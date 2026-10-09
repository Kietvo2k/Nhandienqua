import numpy as np
import pytest

from nhandienqua.data.sources import (
    fruits10_class_from_name,
    quality_label,
    threshold_mask,
    is_original_copy,
)


@pytest.mark.parametrize(
    "source,raw,expected",
    [
        ("kg_mango_disease", "Healthy", "good"),
        ("kg_mango_disease", "Anthracnose", "defect"),
        ("kg_mango_disease", "Stem end Rot", "defect"),
        ("hf_banana_guava_quality", 0, "good"),   # Class_A
        ("hf_banana_guava_quality", 1, "good"),   # Class_B
        ("hf_banana_guava_quality", 2, "defect"),  # Defect
        ("hf_dragonfruit_quality", 1, "good"),    # Fresh Dragon Fruit
        ("hf_dragonfruit_quality", 0, "defect"),  # Defect Dragon Fruit
        ("kg_dragonfruit_vn", "Healthy", "good"),
        ("kg_dragonfruit_vn", "Soft Rot", "defect"),
        ("kg_jackfruit_shapes", "perfect", "good"),
        ("kg_jackfruit_shapes", "deformed", "defect"),
        ("kg_jackfruit_shapes", "malformed", "defect"),
        ("gh_sweet_potato_defect", "A-southern_blight", "defect"),
        ("hf_fruits360", 94, "good"),
        ("hf_fruitseg30", "Mango_Bari", "good"),
    ],
)
def test_quality_label(source, raw, expected):
    assert quality_label(source, raw) == expected


def test_quality_label_unknown_source():
    with pytest.raises(KeyError):
        quality_label("khong_co", "x")


@pytest.mark.parametrize(
    "name,expected",
    [
        ("01_Mango.jpg", "mango"),
        ("33_banana.txt", "banana"),
        ("12_Banana.jpg", "banana"),
        ("05_Jackfruit.jpg", "jackfruit"),
        ("01_ Apple.jpg", None),
        ("07_HogPlum.jpg", None),
    ],
)
def test_fruits10_class_from_name(name, expected):
    assert fruits10_class_from_name(name) == expected


def test_threshold_mask_white_background():
    img = np.full((20, 20, 3), 250, np.uint8)
    img[5:15, 4:12] = (180, 60, 40)
    m = threshold_mask(img, background="white")
    assert m[10, 8] and not m[0, 0]
    assert m.sum() == 10 * 8


def test_threshold_mask_black_background():
    img = np.full((20, 20, 3), 8, np.uint8)
    img[2:6, 2:18] = (170, 140, 120)
    m = threshold_mask(img, background="black")
    assert m.sum() == 4 * 16


def test_is_original_copy():
    assert is_original_copy("Healthy_001_rot0.jpeg")
    assert not is_original_copy("Healthy_001_rot90.jpeg")
    assert is_original_copy("IMG_1924.jpg")
