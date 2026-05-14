from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from usv_faults.config import read_yaml, write_yaml
from usv_faults.preprocessing.windowing import WindowingConfig, build_windows_for_trial
from usv_faults.storage.trials import quality_check_trial, read_events, read_manifest


def make_dataset(config_path: Path, out_dir: Path) -> Dict[str, object]:
    config = read_yaml(config_path)
    raw_root = Path(config["raw_trial_root"])
    if not raw_root.is_absolute():
        raw_root = config_path.parent.parent / raw_root
    out_dir.mkdir(parents=True, exist_ok=True)

    trial_dirs = _selected_trial_dirs(raw_root, config)
    if not trial_dirs:
        raise ValueError(f"no raw trial folders found under {raw_root}")

    windowing_config = WindowingConfig(
        window_ms=float(config["windowing"]["window_ms"]),
        stride_ms=float(config["windowing"]["stride_ms"]),
        current_sample_rate_hz=int(config["preprocessing"]["current_sample_rate_hz"]),
        scalar_features=list(config["preprocessing"]["scalar_features"]),
        expected_input_dim=int(config["preprocessing"]["expected_input_dim"]),
    )

    windows_parts: List[pd.DataFrame] = []
    labels_parts: List[pd.DataFrame] = []
    included_trials: List[str] = []
    for trial_dir in trial_dirs:
        report = quality_check_trial(trial_dir)
        if report.status == "rejected":
            continue
        manifest = read_manifest(trial_dir)
        telemetry = pd.read_parquet(trial_dir / "telemetry.parquet")
        events = read_events(trial_dir)
        windows, labels = build_windows_for_trial(telemetry, manifest, events, windowing_config)
        labels["split"] = _split_for_trial(manifest.trial_id, config)
        windows_parts.append(windows)
        labels_parts.append(labels)
        included_trials.append(manifest.trial_id)

    if not windows_parts:
        raise ValueError("no accepted trials were available for dataset generation")

    windows_df = pd.concat(windows_parts, ignore_index=True)
    labels_df = pd.concat(labels_parts, ignore_index=True)
    windows_df.to_parquet(out_dir / "windows.parquet", index=False)
    labels_df.to_parquet(out_dir / "labels.parquet", index=False)

    split_manifest = {
        "strategy": config.get("split", {}).get("strategy", "by_trial"),
        "train": list(config.get("split", {}).get("train", [])),
        "validation": list(config.get("split", {}).get("validation", [])),
        "test": list(config.get("split", {}).get("test", [])),
    }
    write_yaml(out_dir / "split_manifest.yaml", split_manifest)

    manifest = {
        "dataset_id": config["dataset_id"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_type": config.get("source_type", "unknown"),
        "source_trials": included_trials,
        "windowing": config["windowing"],
        "preprocessing": config["preprocessing"],
        "scaling": config.get("scaling", {}),
        "split": split_manifest,
        "artifacts": {
            "windows": "windows.parquet",
            "labels": "labels.parquet",
            "split_manifest": "split_manifest.yaml",
        },
        "status": "active",
    }
    write_yaml(out_dir / "dataset_manifest.yaml", manifest)

    return {
        "dataset_id": config["dataset_id"],
        "trial_count": len(included_trials),
        "window_count": int(len(windows_df)),
        "input_dim": int(len(windows_df.columns)),
        "out_dir": str(out_dir),
    }


def _selected_trial_dirs(raw_root: Path, config: Dict[str, object]) -> List[Path]:
    explicit = list(config.get("source_trials", []))
    if explicit:
        return [raw_root / str(trial_id) for trial_id in explicit]
    return sorted(path for path in raw_root.iterdir() if path.is_dir())


def _split_for_trial(trial_id: str, config: Dict[str, object]) -> str:
    split = config.get("split", {})
    if not isinstance(split, dict):
        return "unspecified"
    for split_name in ("train", "validation", "test"):
        if trial_id in set(split.get(split_name, [])):
            return split_name
    return "unspecified"
