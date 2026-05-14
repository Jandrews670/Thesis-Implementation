from __future__ import annotations

from typing import Dict

import numpy as np


def validation_percentile_threshold(
    validation_errors: np.ndarray,
    target_false_positive_rate: float,
) -> Dict[str, float]:
    if validation_errors.size == 0:
        raise ValueError("validation error array is empty")
    percentile = 100.0 * (1.0 - target_false_positive_rate)
    threshold = float(np.percentile(validation_errors, percentile))
    false_positive_rate = float(np.mean(validation_errors > threshold))
    return {
        "method": "validation_percentile",
        "target_false_positive_rate": float(target_false_positive_rate),
        "percentile": float(percentile),
        "threshold": threshold,
        "validation_false_positive_rate": false_positive_rate,
        "validation_error_mean": float(np.mean(validation_errors)),
        "validation_error_std": float(np.std(validation_errors)),
        "validation_error_min": float(np.min(validation_errors)),
        "validation_error_max": float(np.max(validation_errors)),
    }
