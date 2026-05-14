from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

from usv_faults.models.sdae import SparseDenoisingAutoencoder
from usv_faults.preprocessing.feature_scaling import StandardFeatureScaler


@dataclass
class LatentExtraction:
    frame: pd.DataFrame
    latent_columns: List[str]
    reconstruction_threshold: float
    model_config: Dict[str, object]
    feature_names: List[str]


@dataclass
class WindowInference:
    reconstruction_errors: np.ndarray
    latents: np.ndarray
    threshold: float
    is_anomaly: np.ndarray
    latent_columns: List[str]
    model_config: Dict[str, object]
    feature_names: List[str]


def extract_latent_windows(model_dir: Path, dataset_dir: Path) -> LatentExtraction:
    windows = pd.read_parquet(dataset_dir / "windows.parquet")
    labels = pd.read_parquet(dataset_dir / "labels.parquet")
    if len(windows) != len(labels):
        raise ValueError("windows and labels row counts do not match")

    inference = infer_windows(model_dir, windows)
    reconstruction_errors = inference.reconstruction_errors
    latents = inference.latents

    latent_columns = inference.latent_columns
    latent_frame = pd.DataFrame(latents, columns=latent_columns)
    frame = pd.concat([labels.reset_index(drop=True), latent_frame], axis=1)
    frame.insert(len(labels.columns), "reconstruction_error", reconstruction_errors)
    frame.insert(len(labels.columns) + 1, "is_anomaly", inference.is_anomaly)
    return LatentExtraction(
        frame=frame,
        latent_columns=latent_columns,
        reconstruction_threshold=inference.threshold,
        model_config=inference.model_config,
        feature_names=inference.feature_names,
    )


def infer_windows(model_dir: Path, windows: pd.DataFrame) -> WindowInference:
    model, model_config, feature_names = load_sdae_model(model_dir)
    scaler = StandardFeatureScaler.load(model_dir / "scaler.joblib")
    threshold = _read_threshold(model_dir / "threshold.json")
    if feature_names and list(windows.columns) != feature_names:
        windows = windows[feature_names]

    values = windows.to_numpy(dtype=np.float32)
    scaled = scaler.transform(values)
    reconstruction_errors, latents = _infer(model, scaled)
    latent_columns = [f"latent_{index:04d}" for index in range(latents.shape[1])]
    return WindowInference(
        reconstruction_errors=reconstruction_errors,
        latents=latents,
        threshold=threshold,
        is_anomaly=reconstruction_errors > threshold,
        latent_columns=latent_columns,
        model_config=model_config,
        feature_names=feature_names,
    )


def load_sdae_model(model_dir: Path) -> Tuple[SparseDenoisingAutoencoder, Dict[str, object], List[str]]:
    checkpoint = _torch_load(model_dir / "model.pt")
    model_config = dict(checkpoint["model_config"])
    model = SparseDenoisingAutoencoder(
        input_dim=int(model_config["input_dim"]),
        hidden_dims=[int(value) for value in model_config["hidden_dims"]],
        latent_dim=int(model_config["latent_dim"]),
        hidden_activation=str(model_config.get("hidden_activation", "relu")),
        output_activation=str(model_config.get("output_activation", "sigmoid")),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, model_config, list(checkpoint.get("feature_names", []))


def _infer(model: SparseDenoisingAutoencoder, values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    errors: List[np.ndarray] = []
    latents: List[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(values), 512):
            batch = torch.from_numpy(values[start : start + 512])
            reconstruction, latent = model(batch)
            batch_errors = torch.mean((reconstruction - batch) ** 2, dim=1)
            errors.append(batch_errors.cpu().numpy())
            latents.append(latent.cpu().numpy())
    return (
        np.concatenate(errors).astype(np.float64),
        np.concatenate(latents).astype(np.float64),
    )


def _read_threshold(path: Path) -> float:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return float(data["threshold"])


def _torch_load(path: Path) -> Dict[str, object]:
    try:
        return torch.load(path, map_location=torch.device("cpu"), weights_only=False)
    except TypeError:
        return torch.load(path, map_location=torch.device("cpu"))
