import numpy as np

from nhandienqua.data.compose import (
    compose_detect_image,
    procedural_belt,
    render_quality_sample,
    rotate_rgba,
)


def _fruit(h=60, w=100, color=(220, 150, 40)):
    rgba = np.zeros((h, w, 4), np.uint8)
    rgba[..., :3] = color
    yy, xx = np.mgrid[:h, :w]
    rgba[..., 3] = (((yy - h / 2) / (h / 2)) ** 2 + ((xx - w / 2) / (w / 2)) ** 2 <= 1) * 255
    return rgba


def test_procedural_belt_shape_and_dark():
    rng = np.random.default_rng(0)
    bg = procedural_belt(120, 200, rng, kind="rubber")
    assert bg.shape == (120, 200, 3) and bg.dtype == np.uint8
    assert bg.mean() < 140  # băng cao su tối màu


def test_rotate_rgba_keeps_object():
    rgba = _fruit()
    rot = rotate_rgba(rgba, 45)
    assert rot.shape[2] == 4
    area0 = (rgba[..., 3] > 127).sum()
    area1 = (rot[..., 3] > 127).sum()
    assert abs(area1 - area0) / area0 < 0.05  # xoay không làm mất/đẻ thêm vật


def test_render_quality_sample_contains_fruit_on_belt():
    rng = np.random.default_rng(1)
    bgs = [procedural_belt(400, 400, rng)]
    img = render_quality_sample(_fruit(), bgs, rng, out_size=224)
    assert img.shape == (224, 224, 3)
    center = img[80:144, 80:144].astype(int)
    assert center[..., 0].mean() > 120  # quả màu vàng/cam nằm giữa ảnh


def test_compose_detect_image_labels_valid():
    rng = np.random.default_rng(2)
    bgs = [procedural_belt(700, 700, rng)]
    pool = {0: [_fruit()], 3: [_fruit(120, 200, (120, 160, 40))]}
    img, labels = compose_detect_image(pool, bgs, rng, size=640, n_objects=(3, 6), scale_px={0: (60, 120), 3: (150, 260)})
    assert img.shape == (640, 640, 3)
    assert 1 <= len(labels) <= 6
    for cls, (x1, y1, x2, y2) in labels:
        assert cls in (0, 3)
        assert 0 <= x1 < x2 <= 640 and 0 <= y1 < y2 <= 640
