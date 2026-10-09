"""Nhandienqua — phát hiện quả + phân loại chất lượng."""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
VISION_ROOT = PROJECT_ROOT / "vision"
WEIGHTS_DIR = VISION_ROOT / "weights"

# Giữ cấu hình/cache của Ultralytics trong dự án (không ghi ra ổ C). Phải đặt TRƯỚC khi import ultralytics.
os.environ.setdefault("YOLO_CONFIG_DIR", str(PROJECT_ROOT / ".cache" / "ultralytics"))
os.environ.setdefault("TORCH_HOME", str(PROJECT_ROOT / ".cache" / "torch"))
os.environ.setdefault("UV_CACHE_DIR", str(PROJECT_ROOT / ".cache" / "uv"))
os.environ.setdefault("PIP_CACHE_DIR", str(PROJECT_ROOT / ".cache" / "pip"))
os.environ.setdefault("YOLO_AUTOINSTALL", "false")  # không tự cài gói ra ngoài dự án
