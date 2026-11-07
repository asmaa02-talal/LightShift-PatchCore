import numpy as np
import pytest

from lightshift.data.illumination import (
    adjust_brightness,
    adjust_contrast,
    adjust_gamma,
)


def test_brightness_preserves_shape_and_dtype():
    image = np.full((8, 8, 3), 100, dtype=np.uint8)
    result = adjust_brightness(image, 1.2)

    assert result.shape == image.shape
    assert result.dtype == np.uint8
    assert result.mean() > image.mean()


def test_brightness_factor_one_preserves_image():
    image = np.full((8, 8, 3), 120, dtype=np.uint8)
    result = adjust_brightness(image, 1.0)

    np.testing.assert_array_equal(result, image)


def test_brightness_rejects_negative_factor():
    image = np.zeros((4, 4, 3), dtype=np.uint8)

    with pytest.raises(ValueError):
        adjust_brightness(image, -0.5)


def test_brightness_clips_values():
    image = np.full((4, 4, 3), 200, dtype=np.uint8)
    result = adjust_brightness(image, 2.0)

    assert result.max() == 255
    assert result.min() >= 0


def test_gamma_rejects_zero():
    image = np.zeros((4, 4, 3), dtype=np.uint8)

    with pytest.raises(ValueError):
        adjust_gamma(image, 0)


def test_gamma_rejects_negative_value():
    image = np.zeros((4, 4, 3), dtype=np.uint8)

    with pytest.raises(ValueError):
        adjust_gamma(image, -1.0)


def test_gamma_one_preserves_image():
    image = np.random.default_rng(42).integers(
        0, 256, (8, 8, 3), dtype=np.uint8
    )
    result = adjust_gamma(image, 1.0)

    np.testing.assert_array_equal(result, image)


def test_contrast_preserves_shape_and_dtype():
    image = np.arange(8 * 8 * 3, dtype=np.uint8).reshape(8, 8, 3)
    result = adjust_contrast(image, 1.5)

    assert result.shape == image.shape
    assert result.dtype == np.uint8


def test_contrast_factor_one_preserves_image():
    image = np.random.default_rng(42).integers(
        0, 256, (8, 8, 3), dtype=np.uint8
    )
    result = adjust_contrast(image, 1.0)

    np.testing.assert_array_equal(result, image)


def test_contrast_rejects_negative_factor():
    image = np.zeros((4, 4, 3), dtype=np.uint8)

    with pytest.raises(ValueError):
        adjust_contrast(image, -1.0)


def test_all_transformations_return_valid_uint8_values():
    image = np.random.default_rng(42).integers(
        0, 256, (16, 16, 3), dtype=np.uint8
    )

    results = [
        adjust_brightness(image, 1.5),
        adjust_contrast(image, 1.5),
        adjust_gamma(image, 1.5),
    ]

    for result in results:
        assert result.dtype == np.uint8
        assert result.min() >= 0
        assert result.max() <= 255

def test_color_temperature_zero_preserves_image():
    from lightshift.data.illumination import adjust_color_temperature

    image = np.full((8, 8, 3), 120, dtype=np.uint8)
    result = adjust_color_temperature(image, 0.0)

    np.testing.assert_array_equal(result, image)


def test_color_temperature_warm_increases_red():
    from lightshift.data.illumination import adjust_color_temperature

    image = np.full((8, 8, 3), 120, dtype=np.uint8)
    result = adjust_color_temperature(image, 0.5)

    assert result[..., 0].mean() > image[..., 0].mean()
    assert result[..., 2].mean() < image[..., 2].mean()


def test_color_temperature_cold_increases_blue():
    from lightshift.data.illumination import adjust_color_temperature

    image = np.full((8, 8, 3), 120, dtype=np.uint8)
    result = adjust_color_temperature(image, -0.5)

    assert result[..., 0].mean() < image[..., 0].mean()
    assert result[..., 2].mean() > image[..., 2].mean()


def test_color_temperature_rejects_invalid_values():
    from lightshift.data.illumination import adjust_color_temperature

    image = np.zeros((4, 4, 3), dtype=np.uint8)

    with pytest.raises(ValueError):
        adjust_color_temperature(image, 1.5)


def test_color_temperature_preserves_shape_and_dtype():
    from lightshift.data.illumination import adjust_color_temperature

    image = np.full((8, 8, 3), 120, dtype=np.uint8)
    result = adjust_color_temperature(image, 0.5)

    assert result.shape == image.shape
    assert result.dtype == np.uint8

from lightshift.data.augmented_dataset import IlluminationAugmentedDataset


def make_normal_samples(count=3):
    samples = []

    for index in range(count):
        samples.append({
            "image": np.full((32, 32, 3), 120 + index, dtype=np.uint8),
            "label": 0,
            "mask": np.zeros((32, 32), dtype=np.uint8),
            "path": f"train/good/{index:03d}.png",
            "category": "bottle",
        })

    return samples


def test_augmented_dataset_length():
    dataset = IlluminationAugmentedDataset(
        make_normal_samples(3),
        seed=42,
        augmentations_per_image=1,
    )

    assert len(dataset) == 6


def test_augmented_dataset_preserves_original_images():
    samples = make_normal_samples(3)
    dataset = IlluminationAugmentedDataset(samples, seed=42)

    for index, sample in enumerate(samples):
        result = dataset[index * 2]

        np.testing.assert_array_equal(result["image"], sample["image"])
        assert result["augmentation"] == "original"


def test_augmented_dataset_preserves_shape_and_dtype():
    dataset = IlluminationAugmentedDataset(
        make_normal_samples(3), seed=42
    )

    for index in range(len(dataset)):
        result = dataset[index]

        assert result["image"].shape == (32, 32, 3)
        assert result["image"].dtype == np.uint8


def test_augmented_dataset_masks_remain_zero():
    dataset = IlluminationAugmentedDataset(
        make_normal_samples(3), seed=42
    )

    for index in range(len(dataset)):
        assert np.all(dataset[index]["mask"] == 0)


def test_augmented_dataset_is_reproducible():
    samples = make_normal_samples(3)

    dataset_a = IlluminationAugmentedDataset(samples, seed=42)
    dataset_b = IlluminationAugmentedDataset(samples, seed=42)

    for index in range(len(dataset_a)):
        np.testing.assert_array_equal(
            dataset_a[index]["image"],
            dataset_b[index]["image"],
        )