import zipfile

import pytest

from fetch_datasets import download_urls, safe_extract


def test_kaggle_url():
    ds = {"name": "x", "source": "kaggle", "ref": "owner/slug"}
    assert download_urls(ds) == [("https://www.kaggle.com/api/v1/datasets/download/owner/slug", "x.zip")]


def test_hf_urls_one_per_file():
    ds = {"name": "y", "source": "hf", "ref": "Org/ds", "files": ["raw/a.parquet", "raw/b.parquet"]}
    assert download_urls(ds) == [
        ("https://huggingface.co/datasets/Org/ds/resolve/main/raw/a.parquet", "raw/a.parquet"),
        ("https://huggingface.co/datasets/Org/ds/resolve/main/raw/b.parquet", "raw/b.parquet"),
    ]


def test_url_source_is_zip():
    ds = {"name": "g", "source": "url", "ref": "https://codeload.github.com/o/r/zip/refs/heads/main"}
    assert download_urls(ds) == [("https://codeload.github.com/o/r/zip/refs/heads/main", "g.zip")]


def test_unknown_source_raises():
    with pytest.raises(ValueError):
        download_urls({"name": "z", "source": "ftp", "ref": "a"})


def test_safe_extract_normal(tmp_path):
    z = tmp_path / "ok.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("dir/a.txt", "hello")
    out = tmp_path / "out"
    safe_extract(z, out)
    assert (out / "dir" / "a.txt").read_text() == "hello"


def test_safe_extract_member_filter(tmp_path):
    z = tmp_path / "big.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("train/sweetpotato/a.jpg", "a")
        zf.writestr("train/apple/b.jpg", "b")
    out = tmp_path / "out"
    safe_extract(z, out, member_filter="/sweetpotato/")
    assert (out / "train" / "sweetpotato" / "a.jpg").exists()
    assert not (out / "train" / "apple").exists()


def test_safe_extract_rejects_path_traversal(tmp_path):
    z = tmp_path / "evil.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("../evil.txt", "x")
    with pytest.raises(ValueError):
        safe_extract(z, tmp_path / "out")
    assert not (tmp_path / "evil.txt").exists()
