from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional
import warnings

import hdbscan
import numpy as np


@dataclass
class ClusterResult:
    labels: np.ndarray
    probabilities: np.ndarray
    persistence: np.ndarray
    method: str = "hdbscan.HDBSCAN"
    details: Dict[str, object] = field(default_factory=dict)


def cluster_latents(latents: np.ndarray, config: Dict[str, object]) -> ClusterResult:
    values = np.asarray(latents, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("latents must be a 2D array")
    if values.shape[0] == 0:
        return ClusterResult(
            labels=np.asarray([], dtype=np.int64),
            probabilities=np.asarray([], dtype=np.float64),
            persistence=np.asarray([], dtype=np.float64),
            details={"sample_count": 0},
        )

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=int(config.get("min_cluster_size", 15)),
        min_samples=int(config.get("min_samples", 15)),
        metric=str(config.get("metric", "euclidean")),
        cluster_selection_method=str(config.get("cluster_selection_method", "eom")),
        allow_single_cluster=bool(config.get("allow_single_cluster", False)),
        prediction_data=bool(config.get("prediction_data", False)),
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="'force_all_finite' was renamed", category=FutureWarning)
        labels = clusterer.fit_predict(values)
    probabilities = getattr(clusterer, "probabilities_", np.ones(values.shape[0], dtype=np.float64))
    persistence = getattr(clusterer, "cluster_persistence_", np.asarray([], dtype=np.float64))
    return ClusterResult(
        labels=np.asarray(labels, dtype=np.int64),
        probabilities=np.asarray(probabilities, dtype=np.float64),
        persistence=np.asarray(persistence, dtype=np.float64),
        details={
            "sample_count": int(values.shape[0]),
            "min_cluster_size": int(config.get("min_cluster_size", 15)),
            "min_samples": int(config.get("min_samples", 15)),
            "metric": str(config.get("metric", "euclidean")),
            "cluster_selection_method": str(config.get("cluster_selection_method", "eom")),
            "allow_single_cluster": bool(config.get("allow_single_cluster", False)),
        },
    )


def cluster_persistence_for_label(result: ClusterResult, label: int) -> Optional[float]:
    if label < 0 or label >= len(result.persistence):
        return None
    return float(result.persistence[label])
