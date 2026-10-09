"""Tải dataset public liệt kê trong configs/datasets.yaml về data/raw/<name>/.

Ví dụ:
    python scripts/fetch_datasets.py --list
    python scripts/fetch_datasets.py                 # tải tất cả
    python scripts/fetch_datasets.py --only kg_mango_disease hf_fruitseg30
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

import requests
import yaml
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "datasets.yaml"
RAW = ROOT / "data" / "raw"


def download_urls(ds: dict) -> list[tuple[str, str]]:
    """Trả về danh sách (url, đường dẫn file tương đối trong thư mục dataset)."""
    src, ref, name = ds["source"], ds["ref"], ds["name"]
    if src == "kaggle":
        return [(f"https://www.kaggle.com/api/v1/datasets/download/{ref}", f"{name}.zip")]
    if src == "hf":
        return [(f"https://huggingface.co/datasets/{ref}/resolve/main/{f}", f) for f in ds["files"]]
    if src == "url":  # link zip trực tiếp, vd. codeload GitHub
        return [(ref, f"{name}.zip")]
    raise ValueError(f"Nguồn không hỗ trợ: {src}")


def safe_extract(zip_path: Path, dest: Path, member_filter: str | None = None) -> None:
    """Giải nén, từ chối mọi entry thoát ra ngoài `dest` (zip-slip). `member_filter`: chỉ lấy entry chứa chuỗi này."""
    dest = dest.resolve()
    with zipfile.ZipFile(zip_path) as zf:
        members = [m for m in zf.namelist() if member_filter is None or member_filter.lower() in m.lower()]
        for member in members:
            target = (dest / member).resolve()
            if dest != target and dest not in target.parents:
                raise ValueError(f"Entry nguy hiểm trong zip: {member}")
        zf.extractall(dest, members=members)


def _stream(url: str, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".part")
    with requests.get(url, stream=True, timeout=60, allow_redirects=True) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(tmp, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=out.name) as bar:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
                bar.update(len(chunk))
    tmp.replace(out)


def fetch(ds: dict, raw: Path = RAW) -> Path:
    dest = raw / ds["name"]
    done = dest / ".done"
    if done.exists():
        print(f"[bỏ qua] {ds['name']} đã có")
        return dest
    for url, rel in download_urls(ds):
        out = dest / rel
        _stream(url, out)
        if out.suffix == ".zip":
            safe_extract(out, dest, ds.get("extract_filter"))
            out.unlink()
    done.write_text(ds["ref"], encoding="utf-8")
    return dest


def load_catalog(path: Path = CONFIG) -> list[dict]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))["datasets"]


def main(argv: list[str] | None = None) -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # console Windows mặc định cp1252
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true", help="chỉ in danh mục")
    ap.add_argument("--only", nargs="*", help="chỉ tải các dataset có tên này")
    args = ap.parse_args(argv)

    catalog = load_catalog()
    if args.only:
        unknown = set(args.only) - {d["name"] for d in catalog}
        if unknown:
            print(f"Không có trong danh mục: {sorted(unknown)}", file=sys.stderr)
            return 2
        catalog = [d for d in catalog if d["name"] in args.only]

    for ds in catalog:
        if args.list:
            print(f"{ds['name']:<26} {ds['use']:<10} {ds['license']:<14} {ds['source']}:{ds['ref']}")
            continue
        try:
            fetch(ds)
        except Exception as e:  # tiếp tục các dataset khác, báo lỗi cuối
            print(f"[LỖI] {ds['name']}: {e}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
