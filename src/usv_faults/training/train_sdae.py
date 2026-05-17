from __future__ import annotations

import csv
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from usv_faults.config import read_yaml, write_yaml
from usv_faults.models.sdae import SparseDenoisingAutoencoder
from usv_faults.performance import PerformanceSampler, sdae_compute_estimates
from usv_faults.preprocessing.feature_scaling import StandardFeatureScaler
from usv_faults.training.simple_plots import write_histogram_png, write_line_plot_png
from usv_faults.training.threshold_search import validation_percentile_threshold


def train_sdae(dataset_dir: Path, config_path: Path, out_dir: Path) -> Dict[str, object]:
    performance_sampler = PerformanceSampler().start()
    config = read_yaml(config_path)
    dataset_manifest = read_yaml(dataset_dir / "dataset_manifest.yaml")
    windows = pd.read_parquet(dataset_dir / "windows.parquet")
    labels = pd.read_parquet(dataset_dir / "labels.parquet")
    if len(windows) != len(labels):
        raise ValueError("windows and labels row counts do not match")

    model_config = config["model"]
    input_dim = int(model_config["input_dim"])
    if len(windows.columns) != input_dim:
        raise ValueError(f"dataset has {len(windows.columns)} features but model expects {input_dim}")

    train_mask = (labels["split"] == "train") & (~labels["is_fault"].astype(bool))
    validation_mask = (labels["split"] == "validation") & (~labels["is_fault"].astype(bool))
    if int(train_mask.sum()) == 0:
        raise ValueError("no healthy training windows found")
    validation_fallback = False
    if int(validation_mask.sum()) == 0:
        validation_mask = train_mask.copy()
        validation_fallback = True

    feature_names = list(windows.columns)
    values = windows.to_numpy(dtype=np.float32)
    scaler = StandardFeatureScaler.fit(values[train_mask.to_numpy()], feature_names)
    scaled = scaler.transform(values)
    train_values = scaled[train_mask.to_numpy()]
    validation_values = scaled[validation_mask.to_numpy()]

    seed = int(config.get("training", {}).get("seed", 20260514))
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cpu")

    model = SparseDenoisingAutoencoder(
        input_dim=input_dim,
        hidden_dims=[int(value) for value in model_config["hidden_dims"]],
        latent_dim=int(model_config["latent_dim"]),
        hidden_activation=str(model_config.get("hidden_activation", "relu")),
        output_activation=str(model_config.get("output_activation", "sigmoid")),
    ).to(device)

    training_config = config["training"]
    optimizer = torch.optim.Adam(model.parameters(), lr=float(training_config["learning_rate"]))
    batch_size = int(training_config["batch_size"])
    epochs = int(training_config["epochs"])
    masking_noise = float(model_config.get("masking_noise", 0.0))
    l1_lambda = float(model_config.get("l1_lambda", 0.0))

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(train_values)),
        batch_size=batch_size,
        shuffle=True,
    )

    history: List[Dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_losses: List[float] = []
        epoch_reconstruction: List[float] = []
        epoch_l1: List[float] = []
        for (batch,) in train_loader:
            batch = batch.to(device)
            corrupted = _apply_masking_noise(batch, masking_noise)
            optimizer.zero_grad()
            reconstruction, latent = model(corrupted)
            reconstruction_loss = torch.mean((reconstruction - batch) ** 2)
            l1_loss = torch.mean(torch.abs(latent))
            loss = reconstruction_loss + l1_lambda * l1_loss
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.item()))
            epoch_reconstruction.append(float(reconstruction_loss.item()))
            epoch_l1.append(float(l1_loss.item()))
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": float(np.mean(epoch_losses)),
                "reconstruction_loss": float(np.mean(epoch_reconstruction)),
                "l1_activation": float(np.mean(epoch_l1)),
            }
        )

    train_errors = reconstruction_errors(model, train_values)
    validation_errors = reconstruction_errors(model, validation_values)
    threshold = validation_percentile_threshold(
        validation_errors,
        float(config["threshold"]["target_false_positive_rate"]),
    )
    all_errors = reconstruction_errors(model, scaled)
    metrics = _metrics(labels, all_errors, train_mask.to_numpy(), validation_mask.to_numpy(), threshold)
    metrics["validation_fallback_used"] = validation_fallback
    metrics["loss_decreased"] = bool(history[-1]["train_loss"] <= history[0]["train_loss"])
    performance_stats = performance_sampler.stop()
    compute_estimates = sdae_compute_estimates(
        model_config,
        train_windows=int(train_mask.sum()),
        epochs=epochs,
    )
    metrics["performance"] = {
        "scope": "train_sdae",
        **performance_stats,
        **compute_estimates,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    model_path = out_dir / "model.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "model_config": model_config,
            "dataset_id": dataset_manifest.get("dataset_id"),
            "feature_names": feature_names,
        },
        model_path,
    )
    scaler.save(out_dir / "scaler.joblib")
    _write_json(out_dir / "threshold.json", threshold)
    _write_json(out_dir / "metrics.json", metrics)
    _write_history(out_dir / "training_history.csv", history)
    shutil.copyfile(config_path, out_dir / "config.yaml")
    write_line_plot_png(plots_dir / "loss_curve.png", [item["train_loss"] for item in history])
    write_histogram_png(plots_dir / "reconstruction_error_hist.png", validation_errors.tolist())

    run_manifest = {
        "run_id": out_dir.name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_type": "baseline_sdae",
        "dataset_id": dataset_manifest.get("dataset_id"),
        "config_file": str(config_path),
        "model": model_config,
        "training": training_config,
        "threshold": threshold,
        "artifacts": {
            "model": "model.pt",
            "scaler": "scaler.joblib",
            "threshold": "threshold.json",
            "training_history": "training_history.csv",
            "metrics": "metrics.json",
            "loss_curve": "plots/loss_curve.png",
            "reconstruction_error_hist": "plots/reconstruction_error_hist.png",
        },
    }
    write_yaml(out_dir / "run_manifest.yaml", run_manifest)

    return {
        "run_id": out_dir.name,
        "dataset_id": dataset_manifest.get("dataset_id"),
        "epochs": epochs,
        "train_windows": int(train_mask.sum()),
        "validation_windows": int(validation_mask.sum()),
        "threshold": threshold["threshold"],
        "out_dir": str(out_dir),
        "loss_decreased": metrics["loss_decreased"],
        "training_wall_time_s": performance_stats["wall_time_s"],
        "training_peak_rss_mb": performance_stats["peak_rss_mb"],
    }


def reconstruction_errors(model: SparseDenoisingAutoencoder, values: np.ndarray) -> np.ndarray:
    model.eval()
    errors: List[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(values), 512):
            batch = torch.from_numpy(values[start : start + 512])
            reconstruction, _latent = model(batch)
            batch_errors = torch.mean((reconstruction - batch) ** 2, dim=1)
            errors.append(batch_errors.cpu().numpy())
    return np.concatenate(errors).astype(np.float64)


def _apply_masking_noise(batch: torch.Tensor, masking_noise: float) -> torch.Tensor:
    if masking_noise <= 0.0:
        return batch
    keep_probability = 1.0 - masking_noise
    mask = torch.bernoulli(torch.full_like(batch, keep_probability))
    return batch * mask


def _metrics(
    labels: pd.DataFrame,
    all_errors: np.ndarray,
    train_mask: np.ndarray,
    validation_mask: np.ndarray,
    threshold: Dict[str, float],
) -> Dict[str, object]:
    is_fault = labels["is_fault"].astype(bool).to_numpy()
    healthy_mask = ~is_fault
    fault_mask = is_fault
    anomaly_mask = all_errors > threshold["threshold"]
    metrics: Dict[str, object] = {
        "train_reconstruction_error_mean": float(np.mean(all_errors[train_mask])),
        "validation_reconstruction_error_mean": float(np.mean(all_errors[validation_mask])),
        "validation_false_positive_rate": float(np.mean(anomaly_mask[validation_mask])),
        "healthy_false_positive_rate": float(np.mean(anomaly_mask[healthy_mask]))
        if int(healthy_mask.sum()) > 0
        else None,
        "true_fault_detection_rate": float(np.mean(anomaly_mask[fault_mask]))
        if int(fault_mask.sum()) > 0
        else None,
        "fault_reconstruction_error_mean": float(np.mean(all_errors[fault_mask]))
        if int(fault_mask.sum()) > 0
        else None,
        "fault_error_greater_than_train_error": bool(np.mean(all_errors[fault_mask]) > np.mean(all_errors[train_mask]))
        if int(fault_mask.sum()) > 0
        else None,
        "threshold": float(threshold["threshold"]),
        "window_count": int(len(labels)),
        "fault_window_count": int(fault_mask.sum()),
        "healthy_window_count": int(healthy_mask.sum()),
    }
    by_fault: Dict[str, Dict[str, float]] = {}
    for fault_label in sorted(set(labels["fault_label"])):
        fault_label_mask = labels["fault_label"].to_numpy() == fault_label
        by_fault[str(fault_label)] = {
            "window_count": int(fault_label_mask.sum()),
            "mean_reconstruction_error": float(np.mean(all_errors[fault_label_mask])),
            "anomaly_rate": float(np.mean(anomaly_mask[fault_label_mask])),
        }
    metrics["by_fault_label"] = by_fault
    return metrics


def _write_json(path: Path, data: Dict[str, object]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


def _write_history(path: Path, history: List[Dict[str, float]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "train_loss", "reconstruction_loss", "l1_activation"])
        writer.writeheader()
        for row in history:
            writer.writerow(row)
