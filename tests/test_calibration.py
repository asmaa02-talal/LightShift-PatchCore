import numpy as np
import pytest

from lightshift.models.conformal_calibration import conformal_threshold


def test_conformal_threshold_returns_infinity_for_too_small_sample():
    scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    assert np.isinf(conformal_threshold(scores, alpha=0.1))


def test_conformal_threshold_rejects_invalid_alpha():
    with pytest.raises(ValueError):
        conformal_threshold(np.array([0.1, 0.2]), alpha=1.0)
