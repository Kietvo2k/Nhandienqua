import numpy as np
import pytest
import torch
from torch import nn

from nhandienqua import WEIGHTS_DIR
from nhandienqua.explain import CamClassifier, defect_map, detection_heatmap


class _Head(nn.Module):
    """Giống Ultralytics Classify: conv → pool → linear."""

    def __init__(self, c1=4, c_=6, nc=3):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(c1, c_, 1, bias=False), nn.SiLU())
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(0.0)
        self.linear = nn.Linear(c_, nc)


class _Fake(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.ModuleList([nn.Conv2d(3, 4, 3, stride=4, padding=1), _Head()])


def test_cam_mean_equals_logits_on_toy_model():
    torch.manual_seed(0)
    fake = _Fake().eval()
    cam = CamClassifier(fake).eval()
    x = torch.rand(2, 3, 32, 32)
    probs, cams = cam(x)
    feat = fake.model[0](x)
    head = fake.model[1]
    logits = head.linear(head.pool(head.conv(feat)).flatten(1))
    assert cams.shape == (2, 3, 8, 8)
    torch.testing.assert_close(cams.mean((2, 3)), logits, rtol=1e-5, atol=1e-5)  # CAM chính xác, không xấp xỉ
    torch.testing.assert_close(probs, logits.softmax(1))


@pytest.mark.skipif(not (WEIGHTS_DIR / "fruit_quality.pt").exists(), reason="chưa có trọng số")
def test_cam_matches_real_quality_model():
    from nhandienqua.yolo import YOLO

    m = YOLO(str(WEIGHTS_DIR / "fruit_quality.pt")).model.eval().float()
    x = torch.rand(1, 3, 224, 224)
    with torch.no_grad():
        ref = m(x)
        ref = ref[0] if isinstance(ref, (tuple, list)) else ref
        probs, cams = CamClassifier(m).eval()(x)
    torch.testing.assert_close(probs, ref, rtol=1e-4, atol=1e-5)
    assert cams.shape[:2] == (1, 10)


def test_defect_map_is_defect_minus_good_of_that_fruit():
    names = ["banana__defect", "banana__good", "mango__defect", "mango__good"]
    cams = np.zeros((4, 2, 2), np.float32)
    cams[2] = [[3, 0], [0, 0]]  # mango defect
    cams[3] = [[1, 1], [1, 1]]  # mango good
    cams[0] = 100  # lớp của quả khác — phải bỏ qua
    m = defect_map(cams, names, "mango")
    assert m.tolist() == [[2, -1], [-1, -1]]


def test_defect_map_unknown_fruit_is_none():
    assert defect_map(np.zeros((2, 2, 2)), ["a__good", "a__defect"], "mango") is None


def test_letterbox_like_ultralytics():
    from nhandienqua.explain import letterbox

    img = np.full((720, 1280, 3), 200, np.uint8)
    out, scale, (px, py) = letterbox(img, 640)
    assert out.shape == (640, 640, 3)
    assert scale == 0.5 and (px, py) == (0, 140)
    assert out[0, 0].tolist() == [114, 114, 114] and out[320, 320].tolist() == [200, 200, 200]


def test_detection_heatmap_grids_and_upsampling():
    # 2 lớp; lưới stride 8 = 2×2 và stride 16 = 1×1 → N = 5 anchor, thứ tự hàng-trước như Ultralytics
    out = np.zeros((1, 6, 5), np.float32)
    out[0, 4, 1] = 0.9  # anchor (y=0, x=1) của lưới 2×2, lớp 0
    out[0, 5, 4] = 0.3  # anchor duy nhất của lưới 1×1, lớp 1 → phủ toàn ảnh sau khi phóng
    h = detection_heatmap(out, grids=[(2, 2), (1, 1)])
    assert h.shape == (2, 2)
    assert h[0, 1] == pytest.approx(0.9)
    assert h[1, 0] == pytest.approx(0.3)  # lấy max giữa các tầng
