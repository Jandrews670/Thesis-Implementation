from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import hdbscan
import numpy as np
import pandas as pd

from usv_faults.clustering.fault_dictionary import decide_latent, load_fault_dictionary
from usv_faults.clustering.latent import extract_latent_windows
from usv_faults.clustering.mahalanobis import squared_mahalanobis
from usv_faults.config import read_yaml


def evaluate_pipeline(model_dir: Path, dictionary_dir: Path, dataset_dir: Path, out_dir: Path) -> Dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    dictionary = load_fault_dictionary(dictionary_dir)
    dictionary_manifest = read_yaml(dictionary_dir / "dictionary_manifest.yaml")
    model_manifest = read_yaml(model_dir / "run_manifest.yaml")
    dataset_manifest = read_yaml(dataset_dir / "dataset_manifest.yaml")
    extraction = extract_latent_windows(model_dir, dataset_dir)
    frame = extraction.frame.copy()

    decisions = _decisions_for_frame(frame, extraction.latent_columns, dictionary)
    decisions.to_csv(out_dir / "poc_window_decisions.csv", index=False)

    dbcv_score, dbcv_status = _dbcv_from_cluster_artifact(dictionary_dir)
    detection_rows = _detection_metrics(frame, model_dir)
    isolation_rows = _isolation_metrics(decisions, dictionary, dbcv_score, dbcv_status)
    cross_domain_rows = _cross_domain_metrics(decisions, extraction.latent_columns, dictionary)

    _write_csv(out_dir / "poc_detection_metrics.csv", detection_rows)
    _write_csv(out_dir / "poc_isolation_metrics.csv", isolation_rows)
    _write_csv(out_dir / "poc_cross_domain_metrics.csv", cross_domain_rows)

    summary = _summary(
        model_dir=model_dir,
        dictionary_dir=dictionary_dir,
        dataset_dir=dataset_dir,
        out_dir=out_dir,
        model_manifest=model_manifest,
        dataset_manifest=dataset_manifest,
        dictionary_manifest=dictionary_manifest,
        detection_rows=detection_rows,
        isolation_rows=isolation_rows,
        cross_domain_rows=cross_domain_rows,
    )
    (out_dir / "poc_summary.md").write_text(summary, encoding="utf-8")

    return {
        "out_dir": str(out_dir),
        "window_count": int(len(frame)),
        "anomaly_count": int(frame["is_anomaly"].astype(bool).sum()),
        "false_positive_rate": _metric_value(detection_rows, "overall", "false_positive_rate"),
        "true_fault_detection_rate": _metric_value(detection_rows, "overall", "true_fault_detection_rate"),
        "true_fault_isolation_rate": _metric_value(isolation_rows, "overall", "true_fault_isolation_rate"),
        "dictionary_id": dictionary.get("dictionary_id", dictionary_dir.name),
    }


def _decisions_for_frame(
    frame: pd.DataFrame,
    latent_columns: List[str],
    dictionary: Dict[str, object],
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for index, row in frame.iterrows():
        latent = row[latent_columns].to_numpy(dtype=np.float64)
        is_anomaly = bool(row["is_anomaly"])
        match = decide_latent(latent, dictionary) if is_anomaly else {"decision": "healthy"}
        rows.append(
            {
                "row_index": int(index),
                "trial_id": row["trial_id"],
                "window_start_s": float(row["window_start_s"]),
                "window_end_s": float(row["window_end_s"]),
                "baseline_id": int(row["baseline_id"]),
                "baseline_name": row["baseline_name"],
                "fault_label": row["fault_label"],
                "is_fault": bool(row["is_fault"]),
                "reconstruction_error": float(row["reconstruction_error"]),
                "is_anomaly": is_anomaly,
                "dictionary_decision": match.get("decision"),
                "matched_fault_id": match.get("fault_id"),
                "matched_fault_label": match.get("label"),
                "matched_cluster_label": match.get("cluster_label"),
                "mahalanobis_distance_sq": match.get("distance"),
                "mahalanobis_threshold": match.get("threshold"),
                **{column: float(row[column]) for column in latent_columns},
            }
        )
    return pd.DataFrame(rows)


def _detection_metrics(frame: pd.DataFrame, model_dir: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    model_size_mb, model_pt_mb = _artifact_sizes(model_dir)
    rows.append(
        {
            "scope": "overall",
            "baseline_id": "",
            "fault_label": "",
            **_detection_rates(frame),
            "model_artifact_size_mb": model_size_mb,
            "model_pt_size_mb": model_pt_mb,
        }
    )
    for baseline_id, group in frame.groupby("baseline_id"):
        rows.append(
            {
                "scope": "baseline",
                "baseline_id": int(baseline_id),
                "fault_label": "",
                **_detection_rates(group),
                "model_artifact_size_mb": model_size_mb,
                "model_pt_size_mb": model_pt_mb,
            }
        )
    for fault_label, group in frame.groupby("fault_label"):
        rows.append(
            {
                "scope": "fault_label",
                "baseline_id": "",
                "fault_label": fault_label,
                **_detection_rates(group),
                "model_artifact_size_mb": model_size_mb,
                "model_pt_size_mb": model_pt_mb,
            }
        )
    return rows


def _detection_rates(frame: pd.DataFrame) -> Dict[str, object]:
    is_fault = frame["is_fault"].astype(bool)
    is_anomaly = frame["is_anomaly"].astype(bool)
    healthy = ~is_fault
    fault = is_fault
    return {
        "window_count": int(len(frame)),
        "healthy_window_count": int(healthy.sum()),
        "fault_window_count": int(fault.sum()),
        "anomaly_count": int(is_anomaly.sum()),
        "false_positive_rate": _mean_or_none(is_anomaly[healthy]),
        "true_fault_detection_rate": _mean_or_none(is_anomaly[fault]),
    }


def _isolation_metrics(
    decisions: pd.DataFrame,
    dictionary: Dict[str, object],
    dbcv_score: Optional[float],
    dbcv_status: str,
) -> List[Dict[str, object]]:
    known_fault_labels = set(str(item) for item in dictionary.get("known_fault_labels", []))
    withheld_fault_labels = set(str(item) for item in dictionary.get("withheld_fault_labels", []))
    fault_anomalies = decisions[decisions["is_fault"].astype(bool) & decisions["is_anomaly"].astype(bool)].copy()
    known_fault_anomalies = fault_anomalies[fault_anomalies["fault_label"].astype(str).isin(known_fault_labels)]
    withheld_anomalies = fault_anomalies[fault_anomalies["fault_label"].astype(str).isin(withheld_fault_labels)]
    correct_known = _correct_known_mask(known_fault_anomalies)
    rows: List[Dict[str, object]] = [
        {
            "scope": "overall",
            "trial_id": "",
            "fault_label": "",
            "fault_anomaly_count": int(len(fault_anomalies)),
            "known_fault_anomaly_count": int(len(known_fault_anomalies)),
            "true_fault_isolation_rate": _mean_or_none(correct_known),
            "withheld_novel_rate": _mean_or_none(withheld_anomalies["dictionary_decision"].eq("novel"))
            if len(withheld_anomalies)
            else None,
            "fault_isolation_latency_s": _overall_latency(decisions, known_fault_labels),
            "dbcv_score": dbcv_score,
            "dbcv_status": dbcv_status,
        }
    ]
    for (trial_id, fault_label), group in fault_anomalies.groupby(["trial_id", "fault_label"]):
        trial_known = group[group["fault_label"].astype(str).isin(known_fault_labels)]
        rows.append(
            {
                "scope": "trial",
                "trial_id": trial_id,
                "fault_label": fault_label,
                "fault_anomaly_count": int(len(group)),
                "known_fault_anomaly_count": int(len(trial_known)),
                "true_fault_isolation_rate": _mean_or_none(_correct_known_mask(trial_known)),
                "withheld_novel_rate": _mean_or_none(group["dictionary_decision"].eq("novel"))
                if str(fault_label) in withheld_fault_labels
                else None,
                "fault_isolation_latency_s": _latency_for_trial(decisions, str(trial_id), str(fault_label), known_fault_labels),
                "dbcv_score": "",
                "dbcv_status": "",
            }
        )
    return rows


def _cross_domain_metrics(
    decisions: pd.DataFrame,
    latent_columns: List[str],
    dictionary: Dict[str, object],
) -> List[Dict[str, object]]:
    known_fault_labels = set(str(item) for item in dictionary.get("known_fault_labels", []))
    rows: List[Dict[str, object]] = []
    for baseline_id in [1, 2, 3, 4]:
        group = decisions[
            decisions["baseline_id"].astype(int).eq(baseline_id)
            & decisions["is_fault"].astype(bool)
            & decisions["is_anomaly"].astype(bool)
            & decisions["fault_label"].astype(str).isin(known_fault_labels)
        ].copy()
        correct = _correct_known_mask(group)
        rows.append(
            {
                "baseline_id": baseline_id,
                "fault_anomaly_count": int(len(group)),
                "cross_domain_accuracy": _mean_or_none(correct),
                "maximum_centroid_drift_distance_sq": _maximum_centroid_drift(group, latent_columns, dictionary),
                "status": "available" if len(group) else "not_available",
            }
        )
    return rows


def _correct_known_mask(frame: pd.DataFrame) -> pd.Series:
    if frame.empty:
        return pd.Series(dtype=bool)
    return frame["dictionary_decision"].eq("known") & (
        frame["fault_label"].astype(str) == frame["matched_fault_label"].astype(str)
    )


def _latency_for_trial(
    decisions: pd.DataFrame,
    trial_id: str,
    fault_label: str,
    known_fault_labels: set,
) -> Optional[float]:
    trial_fault = decisions[
        decisions["trial_id"].astype(str).eq(trial_id)
        & decisions["fault_label"].astype(str).eq(fault_label)
        & decisions["is_fault"].astype(bool)
    ]
    if trial_fault.empty:
        return None
    onset = float(trial_fault["window_start_s"].min())
    if fault_label in known_fault_labels:
        isolated = trial_fault[_correct_known_mask(trial_fault)]
    else:
        isolated = trial_fault[trial_fault["dictionary_decision"].eq("novel")]
    if isolated.empty:
        return None
    return float(max(0.0, float(isolated["window_start_s"].min()) - onset))


def _overall_latency(decisions: pd.DataFrame, known_fault_labels: set) -> Optional[float]:
    latencies: List[float] = []
    for (trial_id, fault_label), _group in decisions[decisions["is_fault"].astype(bool)].groupby(["trial_id", "fault_label"]):
        latency = _latency_for_trial(decisions, str(trial_id), str(fault_label), known_fault_labels)
        if latency is not None:
            latencies.append(latency)
    return float(np.mean(latencies)) if latencies else None


def _maximum_centroid_drift(
    group: pd.DataFrame,
    latent_columns: List[str],
    dictionary: Dict[str, object],
) -> Optional[float]:
    if group.empty:
        return None
    distances: List[float] = []
    for fault_label, label_group in group.groupby("fault_label"):
        entry = _entry_for_label(dictionary, str(fault_label))
        if entry is None:
            continue
        centroid = label_group[latent_columns].to_numpy(dtype=np.float64).mean(axis=0)
        distance = squared_mahalanobis(
            centroid,
            np.asarray(entry["centroid"], dtype=np.float64),
            np.asarray(entry["precision"], dtype=np.float64),
        )
        distances.append(distance)
    return float(max(distances)) if distances else None


def _entry_for_label(dictionary: Dict[str, object], label: str) -> Optional[Dict[str, object]]:
    for entry in dictionary.get("entries", []):
        if str(entry.get("label")) == label:
            return entry
    return None


def _dbcv_from_cluster_artifact(dictionary_dir: Path) -> Tuple[Optional[float], str]:
    assignments_path = dictionary_dir / "cluster_assignments.csv"
    if not assignments_path.exists():
        return None, "not_available_missing_cluster_assignments"
    assignments = pd.read_csv(assignments_path)
    latent_columns = [column for column in assignments.columns if column.startswith("latent_")]
    if not latent_columns or "cluster_label" not in assignments.columns:
        return None, "not_available_missing_latent_or_cluster_columns"
    labels = assignments["cluster_label"].to_numpy(dtype=np.int64)
    non_noise_labels = {int(label) for label in labels if int(label) >= 0}
    if len(non_noise_labels) < 2:
        return None, "not_available_single_cluster"
    values = assignments[latent_columns].to_numpy(dtype=np.float64)
    try:
        return float(hdbscan.validity.validity_index(values, labels)), "available"
    except Exception as exc:
        return None, f"not_available_{exc.__class__.__name__}"


def _summary(
    model_dir: Path,
    dictionary_dir: Path,
    dataset_dir: Path,
    out_dir: Path,
    model_manifest: Dict[str, object],
    dataset_manifest: Dict[str, object],
    dictionary_manifest: Dict[str, object],
    detection_rows: List[Dict[str, object]],
    isolation_rows: List[Dict[str, object]],
    cross_domain_rows: List[Dict[str, object]],
) -> str:
    false_positive_rate = _metric_value(detection_rows, "overall", "false_positive_rate")
    true_fault_detection_rate = _metric_value(detection_rows, "overall", "true_fault_detection_rate")
    isolation_rate = _metric_value(isolation_rows, "overall", "true_fault_isolation_rate")
    cross_available = [row for row in cross_domain_rows if row["status"] == "available"]
    return "\n".join(
        [
            "# Proof-of-Concept Evaluation Summary",
            "",
            f"Created at: {datetime.now(timezone.utc).isoformat()}",
            "",
            "This evaluation uses synthetic proof-of-concept data. It is a software smoke/evidence run, not final thesis evidence.",
            "",
            "## Artifacts",
            "",
            f"- Model: `{model_manifest.get('run_id', model_dir.name)}` at `{model_dir}`",
            f"- Dictionary: `{dictionary_manifest.get('dictionary_id', dictionary_dir.name)}` at `{dictionary_dir}`",
            f"- Dataset: `{dataset_manifest.get('dataset_id', dataset_dir.name)}` at `{dataset_dir}`",
            f"- Report directory: `{out_dir}`",
            "",
            "## Commands",
            "",
            "```powershell",
            f".\\.venv\\Scripts\\python.exe -m usv_faults.cli evaluate --model {model_dir} --dictionary {dictionary_dir} --dataset {dataset_dir} --out {out_dir}",
            "```",
            "",
            "## Metrics",
            "",
            f"- False positive rate: {_format_metric(false_positive_rate)}",
            f"- True fault detection rate: {_format_metric(true_fault_detection_rate)}",
            f"- True fault isolation rate: {_format_metric(isolation_rate)}",
            f"- DBCV: {_format_metric(_metric_value(isolation_rows, 'overall', 'dbcv_score'))} ({_metric_value(isolation_rows, 'overall', 'dbcv_status')})",
            f"- Cross-domain baselines with known fault anomalies: {len(cross_available)} of 4",
            "",
            "## Acceptance Notes",
            "",
            "- SDAE anomaly decisions are computed from the saved reconstruction threshold.",
            "- Known/novel decisions use squared Mahalanobis distance against Ledoit-Wolf dictionary entries.",
            "- Cross-domain metrics are marked `not_available` when the dataset does not contain B1-B4 known fault anomaly windows.",
            "- SWaP-C power/CPU/RAM measurements are not measured in this offline POC command.",
            "",
        ]
    )


def _artifact_sizes(model_dir: Path) -> Tuple[float, float]:
    total_bytes = sum(path.stat().st_size for path in model_dir.rglob("*") if path.is_file())
    model_path = model_dir / "model.pt"
    model_bytes = model_path.stat().st_size if model_path.exists() else 0
    divisor = 1024.0 * 1024.0
    return float(total_bytes / divisor), float(model_bytes / divisor)


def _mean_or_none(values: pd.Series) -> Optional[float]:
    if len(values) == 0:
        return None
    return float(values.astype(float).mean())


def _metric_value(rows: List[Dict[str, object]], scope: str, key: str) -> object:
    for row in rows:
        if row.get("scope") == scope:
            return row.get(key)
    return None


def _format_metric(value: object) -> str:
    if value is None or value == "":
        return "not_available"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
