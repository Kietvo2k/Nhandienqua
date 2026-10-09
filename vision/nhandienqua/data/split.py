"""Chia train/val/test theo NHÓM ảnh gốc để bản augment của cùng một ảnh không rò rỉ sang tập khác."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

_AUG_PREFIX = re.compile(r"^aug_\d+_")
_ROT_SUFFIX = re.compile(r"_rot\d+$")
_ROBOFLOW = re.compile(r"(_(jpe?g|png))?\.rf\.[0-9a-f]+$")


def group_key_from_name(name: str) -> str:
    stem = Path(name).stem.lower()
    stem = _ROBOFLOW.sub("", stem)
    stem = _AUG_PREFIX.sub("", stem)
    return _ROT_SUFFIX.sub("", stem)


def _u(key: str, seed: int) -> float:
    h = hashlib.sha1(f"{seed}:{key}".encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big") / 2**64


def group_split(keys: list[str], val: float = 0.15, test: float = 0.15, seed: int = 0) -> list[str]:
    """Trả về nhãn 'train'/'val'/'test' cho từng phần tử; cùng key → cùng tập."""
    groups = sorted(set(keys), key=lambda k: _u(k, seed))
    n = len(groups)
    n_test = round(n * test)
    n_val = round(n * val)
    if n >= 3:  # dataset nhỏ vẫn phải có val/test
        n_val, n_test = max(1, n_val), max(1, n_test)
    assign = {}
    for i, g in enumerate(groups):
        assign[g] = "test" if i < n_test else "val" if i < n_test + n_val else "train"
    return [assign[k] for k in keys]


def stratified_group_split(keys: list[str], strata: list[str], val: float = 0.15, test: float = 0.15, seed: int = 0) -> list[str]:
    """Như group_split nhưng chia RIÊNG từng tầng (vd. từng lớp) → lớp nhỏ vẫn có val/test."""
    out = [""] * len(keys)
    by: dict[str, list[int]] = {}
    for i, s in enumerate(strata):
        by.setdefault(s, []).append(i)
    for s, idx in by.items():
        for i, sp in zip(idx, group_split([f"{s}|{keys[i]}" for i in idx], val=val, test=test, seed=seed)):
            out[i] = sp
    return out
