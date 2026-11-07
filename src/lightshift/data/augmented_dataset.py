"""Reproducible photometric augmentation for PatchCore training."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from lightshift.data.illumination import (
    adjust_brightness,
    adjust_color_temperature,
    adjust_contrast,
    adjust_gamma,
)


class IlluminationAugmentedDataset:
    """Generate deterministic photometric views of normal training images."""

    def __init__(
        self,
        samples: list[dict[str, Any]],
        seed: int = 42,
        augmentations_per_image: int = 1,
    ) -> None:
        if augmentations_per_image < 0:
            raise ValueError("augmentations_per_image must be non-negative")

        self.samples = samples
        self.seed = seed
        self.augmentations_per_image = augmentations_per_image

        self.transforms: dict[
            str, Callable[[np.ndarray, np.random.Generator], np.ndarray]
        ] = {
            "brightness": self._brightness,
            "contrast": self._contrast,
            "gamma": self._gamma,
            "color_temperature": self._color_temperature,
        }

    def _brightness(
        self, image: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        factor = float(rng.uniform(0.7, 1.3))
        return adjust_brightness(image, factor)

    def _contrast(
        self, image: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        factor = float(rng.uniform(0.7, 1.3))
        return adjust_contrast(image, factor)

    def _gamma(
        self, image: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        gamma = float(rng.uniform(0.7, 1.5))
        return adjust_gamma(image, gamma)

    def _color_temperature(
        self, image: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        temperature = float(rng.uniform(-0.5, 0.5))
        return adjust_color_temperature(image, temperature)

    def __len__(self) -> int:
        return len(self.samples) * (1 + self.augmentations_per_image)

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index < 0 or index >= len(self):
            raise IndexError("Dataset index out of range")

        views_per_image = 1 + self.augmentations_per_image
        sample_index = index // views_per_image
        view_index = index % views_per_image

        sample = self.samples[sample_index]
        image = sample["image"].copy()
        transform_name = "original"

        if view_index > 0:
            rng = np.random.default_rng(
                np.random.SeedSequence([self.seed, sample_index, view_index])
            )

            transform_names = list(self.transforms)
            transform_name = str(rng.choice(transform_names))
            image = self.transforms[transform_name](image, rng)

        return {
            **sample,
            "image": image,
            "mask": np.zeros_like(sample["mask"]),
            "augmentation": transform_name,
        }
