from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from usv_faults.clustering.fault_dictionary import decide_latent, load_fault_dictionary
from usv_faults.clustering.hdbscan_pipeline import cluster_latents
from usv_faults.clustering.latent import infer_windows
from usv_faults.preprocessing.windowing import WindowingConfig, build_windows_for_trial
from usv_faults.storage.trials import read_events, read_manifest


def run_replay_trial(source: str, trial_dir: Path, model_dir: Path, dictionary_dir: Path, out_dir: Path) -> Dict[str, object]:
    if source != "replay":
        raise ValueError("Milestone 5 only implements --source replay")

    out_dir.mkdir(parents=True, exist_ok=True)
    dictionary = load_fault_dictionary(dictionary_dir)
    windows, labels = _windows_for_trial(trial_dir, dictionary)
    inference = infer_windows(model_dir, windows)
    rows = _decision_rows(labels, inference, dictionary)

    trial_id = str(labels["trial_id"].iloc[0]) if len(labels) else trial_dir.name
    out_path = out_dir / f"{trial_id}_replay_decisions.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    return {
        "trial_id": trial_id,
        "out_path": str(out_path),
        "window_count": int(len(rows)),
        "anomaly_count": int(sum(bool(row["is_anomaly"]) for row in rows)),
        "known_count": int(sum(row["dictionary_decision"] == "known" for row in rows)),
        "novel_count": int(sum(row["dictionary_decision"] == "novel" for row in rows)),
    }


def _windows_for_trial(trial_dir: Path, dictionary: Dict[str, object]) -> tuple:
    manifest = read_manifest(trial_dir)
    events = read_events(trial_dir)
    telemetry = pd.read_parquet(trial_dir / "telemetry.parquet")
    preprocessing = dictionary.get("source_dataset_preprocessing", {}) or {}
    windowing = dictionary.get("source_dataset_windowing", {}) or {}
    config = WindowingConfig(
        window_ms=float(windowing.get("window_ms", 100)),
        stride_ms=float(windowing.get("stride_ms", 100)),
        current_sample_rate_hz=int(preprocessing.get("current_sample_rate_hz", 1000)),
        scalar_features=list(preprocessing.get("scalar_features", ["mean", "variance", "peak_to_peak"])),
        expected_input_dim=int(preprocessing.get("expected_input_dim", dictionary.get("input_dim", 2109))),
    )
    return build_windows_for_trial(telemetry, manifest, events, config)


def _decision_rows(labels: pd.DataFrame, inference, dictionary: Dict[str, object]) -> List[Dict[str, object]]:
    rolling_window_size = int(dictionary.get("clustering", {}).get("config", {}).get("rolling_window_size", 300))
    cluster_config = dict(dictionary.get("clustering", {}).get("config", {}))
    cluster_config.pop("sample_count", None)
    min_cluster_size = int(cluster_config.get("min_cluster_size", 15))
    rolling_latents: List[np.ndarray] = []
    rows: List[Dict[str, object]] = []

    for index, label_row in labels.reset_index(drop=True).iterrows():
        latent = inference.latents[index]
        rolling_latents.append(latent)
        if len(rolling_latents) > rolling_window_size:
            rolling_latents = rolling_latents[-rolling_window_size:]

        is_anomaly = bool(inference.is_anomaly[index])
        runtime_cluster_label = -1
        if is_anomaly and len(rolling_latents) >= min_cluster_size:
            runtime_cluster_label = _current_cluster_label(np.asarray(rolling_latents, dtype=np.float64), cluster_config)

        if is_anomaly:
            match = decide_latent(latent, dictionary)
            decision = match.get("decision")
        else:
            match = {}
            decision = "healthy"

        rows.append(
            {
                "timestamp_s": float(label_row["window_end_s"]),
                "trial_id": label_row["trial_id"],
                "reconstruction_error": float(inference.reconstruction_errors[index]),
                "threshold": float(inference.threshold),
                "is_anomaly": is_anomaly,
                "cluster_label": int(runtime_cluster_label),
                "dictionary_decision": decision,
                "matched_fault_id": match.get("fault_id"),
                "matched_fault_label": match.get("label"),
                "mahalanobis_distance_sq": match.get("distance"),
            }
        )
    return rows


def _current_cluster_label(latents: np.ndarray, config: Dict[str, object]) -> int:
    try:
        result = cluster_latents(latents, config)
    except Exception:
        return -1
    if len(result.labels) == 0:
        return -1
    return int(result.labels[-1])

