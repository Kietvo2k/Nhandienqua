"""Đo độ NHẤT QUÁN theo thời gian của pipeline trên video (cùng một quả qua nhiều khung hình).

    python scripts/consistency.py runs/demo/bang_chuyen_test.mp4 [--tta]

Với mỗi track (một quả), xét kết quả TỪNG KHUNG (chưa bỏ phiếu):
- đổi loại quả: tỉ lệ cặp khung liền nhau có nhãn quả khác nhau
- đổi tốt/hư:   tỉ lệ cặp khung liền nhau có quyết định khác nhau
- lệch đa số:   tỉ lệ khung có quyết định khác quyết định đa số của track
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict

import cv2

from nhandienqua import VISION_ROOT
from nhandienqua.models import DEFAULT_DET, DEFAULT_QUALITY, UltralyticsClassifier, UltralyticsDetector
from nhandienqua.pipeline import FruitInspector
from nhandienqua.voting import EmaSmoother


def measure(video: str, tta: bool, det_w: str, cls_w: str, min_frames: int = 5) -> dict:
    det = UltralyticsDetector(det_w, conf=0.25, track=True)
    cls = UltralyticsClassifier(cls_w, tta=tta)
    insp = FruitInspector(det, cls, det.names, cls.names, det_conf=0.35)
    seq: dict[int, list[tuple[str, str, float]]] = defaultdict(list)
    smo: dict[int, list[tuple[str, str, float]]] = defaultdict(list)
    smoother = EmaSmoother()
    cap = cv2.VideoCapture(video)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        ds = insp(frame)
        for d, e in zip(ds, smoother.update(ds)):
            if d.track_id is not None:
                seq[d.track_id].append((d.fruit, d.action, d.p_defect))
                if not e.partial:  # khung "đang vào" không hiển thị quyết định
                    smo[e.track_id].append((e.fruit, e.action, e.p_defect))
    return {"thô (từng khung)": _stats(seq, min_frames),
            "EMA + vùng trễ, bỏ khung chạm mép (hiển thị)": _stats(smo, min_frames)}


def _stats(seq: dict, min_frames: int) -> dict:
    tracks = [s for s in seq.values() if len(s) >= min_frames]
    pairs = sum(len(s) - 1 for s in tracks)
    fruit_flips = sum(a[0] != b[0] for s in tracks for a, b in zip(s, s[1:]))
    act_flips = sum(a[1] != b[1] for s in tracks for a, b in zip(s, s[1:]))
    off_major = sum(sum(x[1] != Counter(y[1] for y in s).most_common(1)[0][0] for x in s) for s in tracks)
    frames = sum(len(s) for s in tracks)
    unstable = sum(any(a[1] != b[1] for a, b in zip(s, s[1:])) for s in tracks)
    return {"tracks": len(tracks), "frames": frames, "fruit_flip": fruit_flips / max(pairs, 1),
            "action_flip": act_flips / max(pairs, 1), "off_majority": off_major / max(frames, 1),
            "unstable_tracks": unstable}


def main(argv: list[str] | None = None) -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--tta", action="store_true", help="trung bình ảnh gốc + lật ngang + lật dọc")
    ap.add_argument("--det", default=str(DEFAULT_DET))
    ap.add_argument("--cls", default=str(DEFAULT_QUALITY))
    a = ap.parse_args(argv)
    res = measure(a.video, a.tta, a.det, a.cls)
    out = VISION_ROOT / "runs" / "eval" / "consistency.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"video": a.video, "tta": a.tta, **res}, ensure_ascii=False, indent=1), encoding="utf-8")
    for mode, r in res.items():
        print(f"[{mode}] TTA={'bật' if a.tta else 'tắt'} · {r['tracks']} track · {r['frames']} khung-quả")
        print(f"  đổi loại quả giữa 2 khung liền: {r['fruit_flip']:.2%}")
        print(f"  đổi tốt/hư giữa 2 khung liền:   {r['action_flip']:.2%}")
        print(f"  khung lệch quyết định đa số:    {r['off_majority']:.2%}")
        print(f"  track có lúc đổi tốt/hư:        {r['unstable_tracks']}/{r['tracks']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
