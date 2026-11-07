"""Transformations photométriques pour LightShift-PatchCore."""

from __future__ import annotations

import numpy as np


def adjust_brightness(image: np.ndarray, factor: float) -> np.ndarray:
    """Modifie la luminosité d'une image uint8 RGB/BGR."""
    if factor < 0:
        raise ValueError("factor doit être positif ou nul")
    result = image.astype(np.float32) * factor
    return np.clip(result, 0, 255).astype(np.uint8)


def adjust_contrast(image: np.ndarray, factor: float) -> np.ndarray:
    """Modifie le contraste autour de la moyenne globale."""
    if factor < 0:
        raise ValueError("factor doit être positif ou nul")
    image_float = image.astype(np.float32)
    mean = image_float.mean(axis=(0, 1), keepdims=True)
    result = (image_float - mean) * factor + mean
    return np.clip(result, 0, 255).astype(np.uint8)


def adjust_gamma(image: np.ndarray, gamma: float) -> np.ndarray:
    """Applique une correction gamma à une image uint8."""
    if gamma <= 0:
        raise ValueError("gamma doit être strictement positif")
    normalized = image.astype(np.float32) / 255.0
    result = np.power(normalized, gamma) * 255.0
    return np.clip(result, 0, 255).astype(np.uint8)


def adjust_color_temperature(
    image: np.ndarray,
    temperature: float,
) -> np.ndarray:
    """Modifie la température de couleur d'une image RGB uint8.

    temperature = 0 : image originale.
    temperature > 0 : tons plus chauds.
    temperature < 0 : tons plus froids.

    La valeur doit appartenir à [-1.0, 1.0].
    """
    if not -1.0 <= temperature <= 1.0:
        raise ValueError("temperature doit être comprise entre -1.0 et 1.0")

    image_float = image.astype(np.float32).copy()

    if temperature == 0:
        return image.copy()

    image_float[..., 0] += temperature * 40.0
    image_float[..., 2] -= temperature * 40.0

    return np.clip(image_float, 0, 255).astype(np.uint8)
