from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
from scipy.stats import chi2
from sklearn.covariance import LedoitWolf


@dataclass
class CovarianceEstimate:
    covariance: np.ndarray
    precision: np.ndarray
    shrinkage: float
    estimator: str
    condition_number: float


def covariance_with_ledoit_wolf(samples: np.ndarray) -> CovarianceEstimate:
    values = np.asarray(samples, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError("samples must be a non-empty 2D array")

    estimator = LedoitWolf().fit(values)
    covariance = np.asarray(estimator.covariance_, dtype=np.float64)
    precision = np.asarray(estimator.precision_, dtype=np.float64)
    return CovarianceEstimate(
        covariance=covariance,
        precision=precision,
        shrinkage=float(estimator.shrinkage_),
        estimator="sklearn.covariance.LedoitWolf",
        condition_number=float(np.linalg.cond(covariance)),
    )


def squared_mahalanobis(vector: np.ndarray, centroid: np.ndarray, precision: np.ndarray) -> float:
    delta = np.asarray(vector, dtype=np.float64) - np.asarray(centroid, dtype=np.float64)
    return float(delta.T @ np.asarray(precision, dtype=np.float64) @ delta)


def chi_square_threshold(latent_dim: int, confidence: float) -> Dict[str, float]:
    if latent_dim <= 0:
        raise ValueError("latent_dim must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1")
    return {
        "threshold": float(chi2.ppf(confidence, latent_dim)),
        "method": "scipy.stats.chi2.ppf",
        "confidence": float(confidence),
        "degrees_of_freedom": int(latent_dim),
    }

