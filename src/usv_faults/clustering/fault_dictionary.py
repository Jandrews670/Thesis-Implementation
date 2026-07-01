from __future__ import annotations

import csv
import json
import os
import tempfile
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from usv_faults.clustering.hdbscan_pipeline import (
    ClusterResult,
    cluster_latents,
    cluster_persistence_for_label,
)
from usv_faults.clustering.latent import extract_latent_windows
from usv_faults.clustering.mahalanobis import (
    chi_square_threshold,
    covariance_with_ledoit_wolf,
    squared_mahalanobis,
)
from usv_faults.config import read_yaml, write_yaml


def build_fault_dictionary(model_dir: Path, dataset_dir: Path, config_path: Path, out_dir: Path) -> Dict[str, object]:
    config = read_yaml(config_path)
    run_manifest = read_yaml(model_dir / "run_manifest.yaml")
    dataset_manifest = read_yaml(dataset_dir / "dataset_manifest.yaml")
    extraction = extract_latent_windows(model_dir, dataset_dir)

    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "cluster_plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    latent_frame = extraction.frame.copy()
    latent_frame.to_parquet(out_dir / "latent_windows.parquet", index=False)

    candidate_mask = _dictionary_candidate_mask(latent_frame, config)
    candidate_frame = latent_frame.loc[candidate_mask].reset_index(drop=True)
    if candidate_frame.empty:
        raise ValueError("no anomaly windows from configured known B0 faults are available for dictionary building")

    candidate_latents = candidate_frame[extraction.latent_columns].to_numpy(dtype=np.float64)
    cluster_result = cluster_latents(candidate_latents, config)
    candidate_frame["cluster_label"] = cluster_result.labels
    candidate_frame["cluster_probability"] = cluster_result.probabilities
    candidate_frame.to_csv(out_dir / "cluster_assignments.csv", index=False)

    latent_dim = len(extraction.latent_columns)
    confidence = float(config.get("mahalanobis_confidence", 0.99))
    threshold = chi_square_threshold(latent_dim, confidence)
    source_model_id = str(run_manifest.get("run_id", model_dir.name))
    source_dataset_id = str(dataset_manifest.get("dataset_id", dataset_dir.name))

    entries, cluster_rows = _dictionary_entries(
        candidate_frame=candidate_frame,
        latent_columns=extraction.latent_columns,
        cluster_result=cluster_result,
        threshold=threshold,
        config=config,
        source_model_id=source_model_id,
        source_dataset_id=source_dataset_id,
    )
    _write_cluster_summary(out_dir / "cluster_summary.csv", cluster_rows)

    decisions = _known_novel_decisions(
        latent_frame=latent_frame,
        latent_columns=extraction.latent_columns,
        entries=entries,
    )
    decisions.to_csv(out_dir / "known_novel_decisions.csv", index=False)

    dictionary = {
        "dictionary_id": out_dir.name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_model_id": source_model_id,
        "source_dataset_id": source_dataset_id,
        "latent_dim": latent_dim,
        "source_dataset_windowing": dataset_manifest.get("windowing", {}),
        "source_dataset_preprocessing": dataset_manifest.get("preprocessing", {}),
        "clustering": {
            "method": cluster_result.method,
            "config": _clustering_details(config, cluster_result),
        },
        "mahalanobis": threshold,
        "known_fault_labels": list(config.get("known_fault_labels", [])),
        "withheld_fault_labels": list(config.get("withheld_fault_labels", [])),
        "entries": entries,
        "decision_summary": _decision_summary(decisions, config),
    }
    _write_json(out_dir / "dictionary.json", dictionary)

    _write_cluster_plot(
        plots_dir / "latent_clusters.png",
        candidate_frame,
        extraction.latent_columns,
        "cluster_label",
        "HDBSCAN clusters in SDAE latent space",
    )
    _write_cluster_plot(
        plots_dir / "latent_fault_labels.png",
        candidate_frame,
        extraction.latent_columns,
        "fault_label",
        "Known fault labels in SDAE latent space",
    )

    manifest = {
        "dictionary_id": out_dir.name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_model": source_model_id,
        "source_dataset": source_dataset_id,
        "config_file": str(config_path),
        "latent_dim": latent_dim,
        "reconstruction_threshold": extraction.reconstruction_threshold,
        "source_dataset_windowing": dataset_manifest.get("windowing", {}),
        "source_dataset_preprocessing": dataset_manifest.get("preprocessing", {}),
        "hdbscan": {
            "method": cluster_result.method,
            "details": _clustering_details(config, cluster_result),
        },
        "mahalanobis": threshold,
        "known_fault_labels": list(config.get("known_fault_labels", [])),
        "withheld_fault_labels": list(config.get("withheld_fault_labels", [])),
        "window_counts": {
            "total": int(len(latent_frame)),
            "anomaly": int(latent_frame["is_anomaly"].astype(bool).sum()),
            "dictionary_candidates": int(len(candidate_frame)),
            "withheld_fault_anomalies": int(_withheld_anomaly_count(latent_frame, config)),
        },
        "cluster_count": int(len([row for row in cluster_rows if int(row["cluster_label"]) >= 0])),
        "dictionary_entry_count": int(len(entries)),
        "dependencies": _dependency_versions(["hdbscan", "scikit-learn", "scipy", "matplotlib"]),
        "artifacts": {
            "dictionary": "dictionary.json",
            "cluster_summary": "cluster_summary.csv",
            "latent_windows": "latent_windows.parquet",
            "cluster_assignments": "cluster_assignments.csv",
            "known_novel_decisions": "known_novel_decisions.csv",
            "cluster_plots": "cluster_plots/",
        },
    }
    write_yaml(out_dir / "dictionary_manifest.yaml", manifest)

    return {
        "dictionary_id": out_dir.name,
        "out_dir": str(out_dir),
        "cluster_count": manifest["cluster_count"],
        "dictionary_entry_count": manifest["dictionary_entry_count"],
        "candidate_window_count": len(candidate_frame),
        "withheld_fault_anomalies": manifest["window_counts"]["withheld_fault_anomalies"],
        "known_fault_match_rate": dictionary["decision_summary"].get("known_fault_match_rate"),
        "withheld_novel_rate": dictionary["decision_summary"].get("withheld_novel_rate"),
    }


def load_fault_dictionary(dictionary_dir: Path) -> Dict[str, object]:
    with (dictionary_dir / "dictionary.json").open("r", encoding="utf-8") as handle:
        return json.load(handle)


def decide_latent(latent: np.ndarray, dictionary: Dict[str, object]) -> Dict[str, object]:
    return _nearest_entry(latent, list(dictionary.get("entries", [])))


def decide_latent_cluster(latents: np.ndarray, dictionary: Dict[str, object]) -> Dict[str, object]:
    values = np.asarray(latents, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0:
        return {"decision": "novel_insufficient_support", "decision_basis": "rolling_cluster_to_dictionary"}
    entries = list(dictionary.get("entries", []))
    if not entries:
        return {"decision": "novel", "decision_basis": "rolling_cluster_to_dictionary"}

    cluster_config = dictionary.get("clustering", {}).get("config", {}) or {}
    required_fraction = float(cluster_config.get("cluster_match_min_member_fraction", 0.50))
    centroid = values.mean(axis=0)
    scored: List[Tuple[float, float, float, float, Dict[str, object]]] = []
    for entry in entries:
        entry_centroid = np.asarray(entry["centroid"], dtype=np.float64)
        entry_precision = np.asarray(entry["precision"], dtype=np.float64)
        centroid_distance = squared_mahalanobis(centroid, entry_centroid, entry_precision)
        threshold = _effective_entry_threshold(entry)
        chi_square_threshold_value = _chi_square_entry_threshold(entry)
        member_distances = np.asarray(
            [squared_mahalanobis(row, entry_centroid, entry_precision) for row in values],
            dtype=np.float64,
        )
        inlier_fraction = float(np.mean(member_distances <= threshold))
        scored.append((centroid_distance, inlier_fraction, threshold, chi_square_threshold_value, entry))

    matching = [
        item for item in scored if item[0] <= item[2] and item[1] >= required_fraction
    ]
    distance, inlier_fraction, threshold, chi_square_threshold_value, entry = min(
        matching or scored,
        key=lambda item: item[0],
    )
    is_known = bool(matching)
    decision = "known" if is_known else "novel"
    if not is_known and distance <= chi_square_threshold_value:
        decision = "novel_empirical_threshold"
    return {
        "decision": decision,
        "decision_basis": "rolling_cluster_to_dictionary",
        "fault_id": entry["fault_id"],
        "label": entry["label"],
        "cluster_label": entry.get("cluster_label"),
        "distance": float(distance),
        "threshold": threshold,
        "chi_square_threshold": chi_square_threshold_value,
        "cluster_support_count": int(values.shape[0]),
        "cluster_member_inlier_fraction": inlier_fraction,
        "cluster_match_min_member_fraction": required_fraction,
    }


def _dictionary_candidate_mask(frame: pd.DataFrame, config: Dict[str, object]) -> pd.Series:
    baseline_id = int(config.get("dictionary_baseline_id", 0))
    fault_labels = frame["fault_label"].astype(str)
    known_fault_labels = set(str(item) for item in config.get("known_fault_labels", []))
    withheld_fault_labels = set(str(item) for item in config.get("withheld_fault_labels", []))
    if known_fault_labels:
        known_mask = fault_labels.isin(known_fault_labels)
    else:
        known_mask = (fault_labels != "none") & (~fault_labels.isin(withheld_fault_labels))
    return (
        frame["is_anomaly"].astype(bool)
        & frame["is_fault"].astype(bool)
        & (frame["baseline_id"].astype(int) == baseline_id)
        & known_mask
    )


def _dictionary_entries(
    candidate_frame: pd.DataFrame,
    latent_columns: List[str],
    cluster_result: ClusterResult,
    threshold: Dict[str, float],
    config: Dict[str, object],
    source_model_id: str,
    source_dataset_id: str,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    entries: List[Dict[str, object]] = []
    cluster_rows: List[Dict[str, object]] = []
    labels = sorted(int(label) for label in set(cluster_result.labels))

    for cluster_label in labels:
        cluster_mask = candidate_frame["cluster_label"].astype(int) == cluster_label
        cluster_frame = candidate_frame.loc[cluster_mask].copy()
        sample_count = int(len(cluster_frame))
        fault_counts = _value_counts(cluster_frame["fault_label"].astype(str))
        trial_ids = sorted(str(item) for item in set(cluster_frame["trial_id"].astype(str)))
        majority_label = _majority_label(fault_counts)
        is_noise = cluster_label < 0
        is_dictionary_entry = not is_noise and sample_count > 0

        cluster_rows.append(
            {
                "cluster_label": cluster_label,
                "sample_count": sample_count,
                "is_dictionary_entry": bool(is_dictionary_entry),
                "majority_fault_label": majority_label or "",
                "fault_label_counts": json.dumps(fault_counts, sort_keys=True),
                "source_trial_ids": ";".join(trial_ids),
                "mean_cluster_probability": float(cluster_frame["cluster_probability"].mean())
                if sample_count
                else 0.0,
                "cluster_persistence": cluster_persistence_for_label(cluster_result, cluster_label),
            }
        )

        if not is_dictionary_entry:
            continue

        values = cluster_frame[latent_columns].to_numpy(dtype=np.float64)
        covariance = covariance_with_ledoit_wolf(values)
        centroid = values.mean(axis=0)
        empirical_threshold = _empirical_mahalanobis_threshold(
            values,
            centroid,
            covariance.precision,
            threshold,
            config,
        )
        entries.append(
            {
                "fault_id": f"fault_{len(entries) + 1:03d}",
                "label": majority_label or f"cluster_{cluster_label}",
                "cluster_label": int(cluster_label),
                "centroid": centroid.tolist(),
                "covariance": covariance.covariance.tolist(),
                "precision": covariance.precision.tolist(),
                "ledoit_wolf_shrinkage": covariance.shrinkage,
                "covariance_estimator": covariance.estimator,
                "covariance_condition_number": covariance.condition_number,
                "sample_count": sample_count,
                "source_trial_ids": trial_ids,
                "source_model_id": source_model_id,
                "source_dataset_id": source_dataset_id,
                "latent_dim": len(latent_columns),
                "mahalanobis_confidence": threshold["confidence"],
                "mahalanobis_threshold": empirical_threshold["effective_threshold"],
                "mahalanobis_effective_threshold": empirical_threshold["effective_threshold"],
                "mahalanobis_chi_square_threshold": threshold["threshold"],
                "mahalanobis_chi_square_method": threshold["method"],
                "mahalanobis_threshold_method": empirical_threshold["effective_method"],
                "mahalanobis_empirical_enabled": empirical_threshold["enabled"],
                "mahalanobis_empirical_threshold": empirical_threshold["empirical_threshold"],
                "mahalanobis_empirical_threshold_uncapped": empirical_threshold[
                    "empirical_threshold_uncapped"
                ],
                "mahalanobis_empirical_percentile": empirical_threshold["percentile"],
                "mahalanobis_empirical_margin": empirical_threshold["margin"],
                "mahalanobis_empirical_min_samples": empirical_threshold["min_samples"],
                "mahalanobis_empirical_status": empirical_threshold["status"],
                "mahalanobis_source_distance_sq_min": empirical_threshold["distance_min"],
                "mahalanobis_source_distance_sq_median": empirical_threshold["distance_median"],
                "mahalanobis_source_distance_sq_p95": empirical_threshold["distance_p95"],
                "mahalanobis_source_distance_sq_p99": empirical_threshold["distance_p99"],
                "mahalanobis_source_distance_sq_max": empirical_threshold["distance_max"],
                "cluster_probability_mean": float(cluster_frame["cluster_probability"].mean()),
                "cluster_persistence": cluster_persistence_for_label(cluster_result, cluster_label),
                "cluster_fault_label_counts": fault_counts,
            }
        )
    return entries, cluster_rows


def _known_novel_decisions(
    latent_frame: pd.DataFrame,
    latent_columns: List[str],
    entries: List[Dict[str, object]],
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    anomaly_frame = latent_frame.loc[latent_frame["is_anomaly"].astype(bool)].copy()
    for _index, row in anomaly_frame.iterrows():
        latent = row[latent_columns].to_numpy(dtype=np.float64)
        match = _nearest_entry(latent, entries)
        rows.append(
            {
                "trial_id": row["trial_id"],
                "window_start_s": float(row["window_start_s"]),
                "window_end_s": float(row["window_end_s"]),
                "fault_label": row["fault_label"],
                "baseline_id": int(row["baseline_id"]),
                "reconstruction_error": float(row["reconstruction_error"]),
                "dictionary_decision": match["decision"],
                "matched_fault_id": match.get("fault_id"),
                "matched_fault_label": match.get("label"),
                "mahalanobis_distance_sq": match.get("distance"),
                "mahalanobis_threshold": match.get("threshold"),
            }
        )
    return pd.DataFrame(rows)


def _nearest_entry(latent: np.ndarray, entries: List[Dict[str, object]]) -> Dict[str, object]:
    if not entries:
        return {"decision": "novel"}
    distances: List[Tuple[float, Dict[str, object]]] = []
    for entry in entries:
        distance = squared_mahalanobis(
            latent,
            np.asarray(entry["centroid"], dtype=np.float64),
            np.asarray(entry["precision"], dtype=np.float64),
        )
        distances.append((distance, entry))
    distance, entry = min(distances, key=lambda item: item[0])
    threshold = _effective_entry_threshold(entry)
    chi_square_threshold_value = _chi_square_entry_threshold(entry)
    decision = "known" if distance <= threshold else "novel"
    if decision != "known" and distance <= chi_square_threshold_value:
        decision = "novel_empirical_threshold"
    return {
        "decision": decision,
        "fault_id": entry["fault_id"],
        "label": entry["label"],
        "cluster_label": entry.get("cluster_label"),
        "distance": float(distance),
        "threshold": threshold,
        "chi_square_threshold": chi_square_threshold_value,
    }


def _effective_entry_threshold(entry: Dict[str, object]) -> float:
    return float(entry.get("mahalanobis_effective_threshold", entry.get("mahalanobis_threshold")))


def _chi_square_entry_threshold(entry: Dict[str, object]) -> float:
    return float(entry.get("mahalanobis_chi_square_threshold", entry.get("mahalanobis_threshold")))


def _empirical_mahalanobis_threshold(
    values: np.ndarray,
    centroid: np.ndarray,
    precision: np.ndarray,
    chi_square: Dict[str, float],
    config: Dict[str, object],
) -> Dict[str, object]:
    distances = np.asarray(
        [squared_mahalanobis(row, centroid, precision) for row in values],
        dtype=np.float64,
    )
    if distances.size == 0:
        raise ValueError("cannot calibrate an empirical threshold without cluster members")

    enabled = bool(config.get("mahalanobis_empirical_enabled", True))
    percentile = float(config.get("mahalanobis_empirical_percentile", 0.95))
    margin = float(config.get("mahalanobis_empirical_margin", 1.0))
    min_samples = int(config.get("mahalanobis_empirical_min_samples", 5))
    if not 0.0 < percentile <= 1.0:
        raise ValueError("mahalanobis_empirical_percentile must be in (0, 1]")
    if margin <= 0.0:
        raise ValueError("mahalanobis_empirical_margin must be positive")
    if min_samples <= 0:
        raise ValueError("mahalanobis_empirical_min_samples must be positive")

    chi_threshold = float(chi_square["threshold"])
    quantile_distance = float(np.quantile(distances, percentile))
    empirical_uncapped = float(quantile_distance * margin)
    empirical_capped = float(min(chi_threshold, empirical_uncapped))
    use_empirical = enabled and int(distances.size) >= min_samples and empirical_uncapped > 0.0
    effective_threshold = empirical_capped if use_empirical else chi_threshold
    status = "used" if use_empirical else ("disabled" if not enabled else "not_enough_samples")

    return {
        "enabled": enabled,
        "status": status,
        "percentile": percentile,
        "margin": margin,
        "min_samples": min_samples,
        "empirical_threshold": empirical_capped if use_empirical else None,
        "empirical_threshold_uncapped": empirical_uncapped if use_empirical else None,
        "effective_threshold": float(effective_threshold),
        "effective_method": (
            "source_cluster_mahalanobis_percentile_margin_capped_by_chi_square"
            if use_empirical
            else chi_square["method"]
        ),
        "distance_min": float(np.min(distances)),
        "distance_median": float(np.median(distances)),
        "distance_p95": float(np.quantile(distances, 0.95)),
        "distance_p99": float(np.quantile(distances, 0.99)),
        "distance_max": float(np.max(distances)),
    }


def _decision_summary(decisions: pd.DataFrame, config: Dict[str, object]) -> Dict[str, Optional[float]]:
    known_fault_labels = set(str(item) for item in config.get("known_fault_labels", []))
    withheld_fault_labels = set(str(item) for item in config.get("withheld_fault_labels", []))
    if decisions.empty:
        return {
            "anomaly_decision_count": 0,
            "known_fault_match_rate": None,
            "withheld_novel_rate": None,
        }
    known_faults = decisions["fault_label"].astype(str).isin(known_fault_labels)
    withheld_faults = decisions["fault_label"].astype(str).isin(withheld_fault_labels)
    known_correct = decisions["dictionary_decision"].eq("known") & (
        decisions["fault_label"].astype(str) == decisions["matched_fault_label"].astype(str)
    )
    return {
        "anomaly_decision_count": int(len(decisions)),
        "known_decision_rate": float(decisions["dictionary_decision"].eq("known").mean()),
        "novel_decision_rate": float(_novel_dictionary_mask(decisions["dictionary_decision"]).mean()),
        "known_fault_match_rate": float(known_correct[known_faults].mean()) if known_faults.any() else None,
        "withheld_novel_rate": float(
            _novel_dictionary_mask(decisions.loc[withheld_faults, "dictionary_decision"]).mean()
        )
        if withheld_faults.any()
        else None,
    }


def _novel_dictionary_mask(values: pd.Series) -> pd.Series:
    return values.astype(str).str.startswith("novel")


def _withheld_anomaly_count(frame: pd.DataFrame, config: Dict[str, object]) -> int:
    withheld_fault_labels = set(str(item) for item in config.get("withheld_fault_labels", []))
    if not withheld_fault_labels:
        return 0
    mask = frame["is_anomaly"].astype(bool) & frame["fault_label"].astype(str).isin(withheld_fault_labels)
    return int(mask.sum())


def _clustering_details(config: Dict[str, object], cluster_result: ClusterResult) -> Dict[str, object]:
    details = dict(cluster_result.details)
    details["rolling_window_size"] = int(config.get("rolling_window_size", 30))
    details["min_runtime_cluster_size"] = int(
        config.get("min_runtime_cluster_size", config.get("min_cluster_size", 15))
    )
    details["cluster_match_min_member_fraction"] = float(
        config.get("cluster_match_min_member_fraction", 0.50)
    )
    details["mahalanobis_empirical_enabled"] = bool(
        config.get("mahalanobis_empirical_enabled", True)
    )
    details["mahalanobis_empirical_percentile"] = float(
        config.get("mahalanobis_empirical_percentile", 0.95)
    )
    details["mahalanobis_empirical_margin"] = float(
        config.get("mahalanobis_empirical_margin", 1.0)
    )
    details["mahalanobis_empirical_min_samples"] = int(
        config.get("mahalanobis_empirical_min_samples", 5)
    )
    details["event_window_size"] = int(config.get("event_window_size", details["rolling_window_size"]))
    details["event_min_anomaly_votes"] = int(config.get("event_min_anomaly_votes", 3))
    details["event_min_anomaly_fraction"] = float(config.get("event_min_anomaly_fraction", 0.30))
    details["event_min_known_votes"] = int(config.get("event_min_known_votes", 3))
    details["event_min_known_fraction"] = float(config.get("event_min_known_fraction", 0.15))
    details["event_min_known_purity"] = float(config.get("event_min_known_purity", 0.50))
    details["event_min_novel_votes"] = int(config.get("event_min_novel_votes", 3))
    details["event_min_novel_fraction"] = float(config.get("event_min_novel_fraction", 0.15))
    return details


def _value_counts(values: Iterable[str]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for value in values:
        counts[str(value)] = counts.get(str(value), 0) + 1
    return counts


def _majority_label(counts: Dict[str, int]) -> Optional[str]:
    if not counts:
        return None
    return max(counts.items(), key=lambda item: (item[1], item[0]))[0]


def _write_cluster_summary(path: Path, rows: List[Dict[str, object]]) -> None:
    fieldnames = [
        "cluster_label",
        "sample_count",
        "is_dictionary_entry",
        "majority_fault_label",
        "fault_label_counts",
        "source_trial_ids",
        "mean_cluster_probability",
        "cluster_persistence",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_cluster_plot(
    path: Path,
    frame: pd.DataFrame,
    latent_columns: List[str],
    color_column: str,
    title: str,
) -> None:
    cache_env = os.environ.get("MPLCONFIGDIR")
    cache_dir = Path(cache_env) if cache_env else Path(tempfile.gettempdir()) / "usv_faults_matplotlib_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(cache_dir)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = frame[latent_columns[0]].to_numpy(dtype=np.float64)
    y = frame[latent_columns[1]].to_numpy(dtype=np.float64) if len(latent_columns) > 1 else np.zeros(len(frame))
    color_codes, labels = pd.factorize(frame[color_column].astype(str))

    figure, axis = plt.subplots(figsize=(7, 5), dpi=120)
    scatter = axis.scatter(x, y, c=color_codes, cmap="tab10", s=28, alpha=0.85)
    axis.set_title(title)
    axis.set_xlabel(latent_columns[0])
    axis.set_ylabel(latent_columns[1] if len(latent_columns) > 1 else "zero")
    handles = scatter.legend_elements()[0]
    axis.legend(handles, labels, title=color_column, loc="best", fontsize=8)
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def _dependency_versions(package_names: List[str]) -> Dict[str, str]:
    versions: Dict[str, str] = {}
    for package_name in package_names:
        try:
            versions[package_name] = metadata.version(package_name)
        except metadata.PackageNotFoundError:
            versions[package_name] = "not_installed"
    return versions


def _write_json(path: Path, data: Dict[str, object]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)
