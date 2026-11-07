"""B3 : calibration conforme de scores normaux."""

from __future__ import annotations

import math
import numpy as np


def conformal_threshold(normal_scores: np.ndarray, alpha: float) -> float:
    """Seuil split-conformal avec convention +inf si le quantile dépasse l'échantillon.

    Pour un score élevé = plus anomal, on prend le quantile d'ordre
    ceil((n+1)*(1-alpha)) des scores de calibration normaux triés par ordre croissant.
    """
    scores = np.asarray(normal_scores, dtype=float).reshape(-1)
    if scores.size == 0:
        raise ValueError("Au moins un score normal de calibration est requis.")
    if not 0 < alpha < 1:
        raise ValueError("alpha doit être compris strictement entre 0 et 1.")
    k = math.ceil((scores.size + 1) * (1 - alpha))
    if k > scores.size:
        return float("inf")
    return float(np.sort(scores)[k - 1])
