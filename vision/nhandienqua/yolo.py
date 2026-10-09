"""Điểm import Ultralytics duy nhất của dự án: ép datasets/weights/runs nằm trong `vision/`."""

from . import VISION_ROOT, WEIGHTS_DIR  # noqa: F401  (đặt YOLO_CONFIG_DIR trước khi import ultralytics)
from ultralytics import YOLO  # noqa: E402
from ultralytics.utils import SETTINGS  # noqa: E402

_WANT = {
    "datasets_dir": str(VISION_ROOT / "data"),
    "weights_dir": str(WEIGHTS_DIR),
    "runs_dir": str(VISION_ROOT / "runs"),
}
if any(SETTINGS.get(k) != v for k, v in _WANT.items()):
    SETTINGS.update(_WANT)

__all__ = ["YOLO", "SETTINGS"]
