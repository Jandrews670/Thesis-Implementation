from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np


@dataclass
class StandardFeatureScaler:
    mean_: np.ndarray
    scale_: np.ndarray
    feature_names: List[str]

    @classmethod
    def fit(cls, values: np.ndarray, feature_names: List[str]) -> "StandardFeatureScaler":
        mean = values.mean(axis=0)
        scale = values.std(axis=0)
        scale[scale == 0.0] = 1.0
        return cls(mean_=mean.astype(np.float32), scale_=scale.astype(np.float32), feature_names=feature_names)

    def transform(self, values: np.ndarray) -> np.ndarray:
        return ((values - self.mean_) / self.scale_).astype(np.float32)

    def fit_transform(cls, values: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Use StandardFeatureScaler.fit(...).transform(...) explicitly.")

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump(
                {
                    "mean": self.mean_,
                    "scale": self.scale_,
                    "feature_names": self.feature_names,
                    "format": "pickle-standard-feature-scaler-v1",
                },
                handle,
            )

    @classmethod
    def load(cls, path: Path) -> "StandardFeatureScaler":
        with path.open("rb") as handle:
            data = pickle.load(handle)
        return cls(
            mean_=np.asarray(data["mean"], dtype=np.float32),
            scale_=np.asarray(data["scale"], dtype=np.float32),
            feature_names=list(data["feature_names"]),
        )
