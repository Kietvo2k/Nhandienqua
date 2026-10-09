import numpy as np
import pytest

from nhandienqua.pipeline import FruitInspector, crop_with_pad, decide_quality
from nhandienqua.voting import TrackVoter

FRUITS = ["mango", "banana", "dragon_fruit", "jackfruit", "sweet_potato"]
QNAMES = ["banana__defect", "banana__good", "mango__defect", "mango__good"]


def test_decide_quality_only_considers_detected_fruit():
    # model nghiêng mạnh về banana__defect, nhưng detector báo xoài → chỉ xét lớp của xoài
    probs = np.array([0.7, 0.1, 0.05, 0.15])
    label, p_defect = decide_quality(probs, QNAMES, "mango", threshold=0.5)
    assert label == "good"
    assert p_defect == pytest.approx(0.05 / 0.20)


def test_decide_quality_threshold():
    probs = np.array([0.0, 0.0, 0.45, 0.55])
    assert decide_quality(probs, QNAMES, "mango", threshold=0.5)[0] == "good"
    assert decide_quality(probs, QNAMES, "mango", threshold=0.4)[0] == "defect"


def test_decide_quality_unknown_fruit_falls_back_to_reject():
    # quả chưa có lớp chất lượng → an toàn: loại ra để người kiểm tra
    label, p = decide_quality(np.array([0.5, 0.5, 0.0, 0.0]), QNAMES, "jackfruit", threshold=0.5)
    assert label == "defect" and p == 1.0


def test_crop_with_pad_clips_to_image():
    img = np.zeros((100, 200, 3), np.uint8)
    c, (x1, y1, x2, y2) = crop_with_pad(img, (190, 90, 210, 120), pad=0.1)
    assert (x1, y1, x2, y2) == (188, 87, 200, 100)
    assert c.shape == (13, 12, 3)


class FakeDetector:
    def __call__(self, frame):
        # (box, lớp, conf, track_id)
        return [((10, 10, 60, 50), 0, 0.9, 1), ((100, 10, 150, 60), 1, 0.8, 2), ((5, 5, 8, 8), 0, 0.2, 3)]


class FakeClassifier:
    def __call__(self, crops):
        out = []
        for c in crops:
            # vùng ảnh sáng → hư (giả)
            out.append([0.0, 0.0, 0.9, 0.1] if c.mean() > 100 else [0.1, 0.9, 0.0, 0.0])
        return np.array(out)


def test_inspector_end_to_end_with_fakes():
    frame = np.zeros((80, 200, 3), np.uint8)
    frame[10:50, 10:60] = 200  # xoài "hư"
    insp = FruitInspector(FakeDetector(), FakeClassifier(), FRUITS, QNAMES, det_conf=0.35)
    ds = insp(frame)
    assert len(ds) == 2  # bbox conf 0.2 bị bỏ
    mango = next(d for d in ds if d.fruit == "mango")
    banana = next(d for d in ds if d.fruit == "banana")
    assert mango.action == "reject" and mango.quality == "defect" and mango.track_id == 1
    assert banana.action == "accept" and banana.quality == "good"
    assert 0 <= mango.p_defect <= 1


def test_predict_in_batches_respects_static_batch():
    # ONNX / TensorRT xuất batch cố định = 1 → không được gửi 3 ảnh cắt một lần (lỗi "Got: 3 Expected: 1")
    from nhandienqua.models import predict_in_batches

    calls = []

    def fake_predict(batch):
        calls.append(len(batch))
        if len(batch) > 1:
            raise RuntimeError("Got: 3 Expected: 1")
        return [np.full(4, float(batch[0][0, 0, 0]))]

    crops = [np.full((5, 5, 3), v, np.uint8) for v in (1, 2, 3)]
    out = predict_in_batches(fake_predict, crops, batch=1)
    assert calls == [1, 1, 1]
    assert out.shape == (3, 4) and out[:, 0].tolist() == [1, 2, 3]
    calls.clear()
    predict_in_batches(lambda b: [np.zeros(4)] * len(b), crops, batch=None)  # .pt: gửi một lần
    assert calls == []


def test_tta_views_average_original_hflip_vflip():
    from nhandienqua.models import tta_views, tta_merge

    crop = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)
    views = tta_views([crop, crop + 1])
    assert len(views) == 6  # mỗi ảnh cắt: gốc, lật ngang, lật dọc
    assert (views[1] == crop[:, ::-1]).all() and (views[2] == crop[::-1]).all()
    probs = np.array([[1.0, 0], [0, 1.0], [0.5, 0.5], [0, 1.0], [0, 1.0], [0, 1.0]])
    assert tta_merge(probs, 2).tolist() == [[0.5, 0.5], [0.0, 1.0]]


def _d(track, x, p, fruit="mango"):
    from nhandienqua.pipeline import Decision

    return Decision(fruit=fruit, box=(x - 10, 0, x + 10, 20), det_conf=0.9, quality="defect" if p >= 0.5 else "good",
                    p_defect=p, action="reject" if p >= 0.5 else "accept", track_id=track)


def test_voter_emits_once_when_crossing_line_with_mean_vote():
    v = TrackVoter(line_x=100, threshold=0.5, min_frames=2)
    assert v.update([_d(1, 40, 0.9)]) == []
    assert v.update([_d(1, 70, 0.8)]) == []
    out = v.update([_d(1, 105, 0.2)])  # qua vạch: TB (0.9+0.8+0.2)/3 = 0.63 → loại
    assert len(out) == 1 and out[0]["action"] == "reject" and out[0]["frames"] == 3
    assert out[0]["p_defect"] == pytest.approx(0.6333, abs=1e-3)
    assert v.update([_d(1, 130, 0.9)]) == []  # không phát lại cho track đã quyết


def test_voter_ignores_untracked_and_handles_many_tracks():
    v = TrackVoter(line_x=50, threshold=0.5, min_frames=1)
    v.update([_d(2, 40, 0.1, "banana")])
    out = v.update([_d(None, 60, 0.9), _d(2, 60, 0.1, "banana"), _d(3, 10, 0.9)])
    assert [(o["track_id"], o["action"], o["fruit"]) for o in out] == [(2, "accept", "banana")]


def test_ema_smoother_stabilizes_borderline_track():
    from nhandienqua.voting import EmaSmoother

    sm = EmaSmoother(alpha=0.3, threshold=0.5)
    raw = [0.6, 0.45, 0.55, 0.48, 0.58, 0.46]  # dao động quanh ngưỡng → thô đổi quyết định 5 lần
    out = [sm.update([_d(1, 10 * i, p)])[0] for i, p in enumerate(raw)]
    acts = [d.action for d in out]
    assert sum(a != b for a, b in zip(acts, acts[1:])) == 0  # sau làm mượt: không nhảy
    assert out[0].p_defect == pytest.approx(0.6)  # khung đầu giữ nguyên
    assert out[1].p_defect == pytest.approx(0.3 * 0.45 + 0.7 * 0.6)


def test_inspector_marks_fruit_cut_by_frame_edge_as_partial():
    frame = np.zeros((80, 200, 3), np.uint8)

    class Det:
        def __call__(self, f):
            return [((0, 10, 40, 50), 0, 0.9, 1), ((100, 10, 150, 60), 1, 0.9, 2), ((170, 5, 200, 40), 1, 0.9, 3)]

    insp = FruitInspector(Det(), FakeClassifier(), FRUITS, QNAMES, det_conf=0.35)
    assert [d.partial for d in insp(frame)] == [True, False, True]


def test_voter_ignores_partial_frames_when_full_views_exist():
    v = TrackVoter(line_x=100, threshold=0.5)
    edge = _d(1, 10, 0.95)
    edge.partial = True  # quả mới vào khung, ảnh cắt thiếu → xác suất không đáng tin
    v.update([edge])
    v.update([_d(1, 60, 0.1)])
    out = v.update([_d(1, 105, 0.1)])
    assert out[0]["action"] == "accept" and out[0]["frames"] == 2  # chỉ tính 2 khung nhìn trọn quả


def test_ema_smoother_hysteresis_band():
    from nhandienqua.voting import EmaSmoother

    sm = EmaSmoother(alpha=1.0, threshold=0.5, band=0.1)  # alpha=1: không EMA, chỉ thử vùng trễ
    acts = [sm.update([_d(1, i, p)])[0].action for i, p in enumerate([0.3, 0.55, 0.65, 0.45, 0.35])]
    # 0.55 chưa đủ 0.6 → vẫn tốt; 0.65 → hư; 0.45 chưa dưới 0.4 → giữ hư; 0.35 → tốt
    assert acts == ["accept", "accept", "reject", "reject", "accept"]


def test_ema_smoother_keeps_tracks_separate_and_passes_untracked():
    from nhandienqua.voting import EmaSmoother

    sm = EmaSmoother(alpha=0.5)
    sm.update([_d(1, 0, 0.9), _d(2, 0, 0.1)])
    a, b, c = sm.update([_d(1, 5, 0.1), _d(2, 5, 0.1), _d(None, 5, 0.8)])
    assert a.p_defect == pytest.approx(0.5) and b.p_defect == pytest.approx(0.1)
    assert c.p_defect == 0.8 and c.action == "reject"


def test_voter_no_decision_for_track_born_past_the_line():
    # tracker đổi ID khi quả ĐÃ qua vạch → ID mới xuất hiện bên phải vạch: không được quyết định lần 2 (gạt nhầm)
    v = TrackVoter(line_x=100, threshold=0.5)
    v.update([_d(1, 80, 0.1)])
    assert len(v.update([_d(1, 105, 0.1)])) == 1
    assert v.update([_d(7, 115, 0.9)]) == []  # ID mới sinh sau vạch
    assert v.update([_d(7, 140, 0.9)]) == []
