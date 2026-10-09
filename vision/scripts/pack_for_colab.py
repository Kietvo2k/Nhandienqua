"""Đóng gói dataset đã dựng để train trên Google Colab.

    python scripts/pack_for_colab.py
→ vision/colab/ndq_quality.zip, vision/colab/ndq_detect.zip (ảnh JPEG đã nén → lưu kiểu STORED cho nhanh)
Tải 2 file zip + vision/colab/train_colab.ipynb lên Google Drive, thư mục MyDrive/nhandienqua/.
"""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

from tqdm import tqdm

from nhandienqua import VISION_ROOT

OUT = VISION_ROOT / "colab"


def pack(src: Path, dst: Path) -> None:
    files = [p for p in src.rglob("*") if p.is_file()]
    with zipfile.ZipFile(dst, "w", compression=zipfile.ZIP_STORED) as zf:
        for p in tqdm(files, desc=dst.name):
            # data.yaml chứa đường dẫn tuyệt đối của máy này → notebook sẽ ghi lại cho Colab
            zf.write(p, p.relative_to(src.parent).as_posix())
    print(f"{dst}  {dst.stat().st_size / 1e6:.0f} MB, {len(files)} file")


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    OUT.mkdir(exist_ok=True)
    for name in ("quality", "detect"):
        src = VISION_ROOT / "data" / name
        if not src.exists():
            print(f"Chưa có {src} — chạy build_{name}_ds.py trước.", file=sys.stderr)
            return 2
        pack(src, OUT / f"ndq_{name}.zip")
    return 0


if __name__ == "__main__":
    sys.exit(main())
