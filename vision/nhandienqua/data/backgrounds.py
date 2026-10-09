"""Nền băng chuyền: ảnh băng thật (dataset josembm, gồm băng trống + có quả mọng làm vật gây nhiễu) + nền sinh thủ tục."""

from __future__ import annotations

import numpy as np

from .compose import procedural_belt
from .sources import RAW, _files, _open_rgb


def load_backgrounds(rng: np.random.Generator, n_procedural: int = 80, max_real: int = 160, side: int = 720) -> list[np.ndarray]:
    import cv2

    bgs: list[np.ndarray] = []
    real = _files(RAW / "kg_conveyor_blueberry")
    empty = [p for p in real if p.parent.name == "Vacio"]
    others = [p for p in real if p.parent.name != "Vacio"]
    rng.shuffle(others)
    for p in empty + others[: max(0, max_real - len(empty))]:
        img = _open_rgb(p)
        s = side / max(img.shape[:2])
        bgs.append(cv2.resize(img, (round(img.shape[1] * s), round(img.shape[0] * s)), interpolation=cv2.INTER_AREA))
    for _ in range(n_procedural):
        bgs.append(procedural_belt(side, side, rng))
    return bgs
