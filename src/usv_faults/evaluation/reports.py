from __future__ import annotations

import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import hdbscan
import numpy as np
import pandas as pd
import torch

from usv_faults.clustering.fault_dictionary import decide_latent, load_fault_dictionary
from usv_faults.clustering.latent import extract_latent_windows, load_sdae_model
from usv_faults.clustering.mahalanobis import squared_mahalanobis
from usv_faults.config import read_yaml
from usv_faults.performance import PerformanceSampler, sdae_compute_estimates
from usv_faults.preprocessing.feature_scaling import StandardFeatureScaler


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
    performance_rows = _performance_metrics(model_dir, dataset_dir)

    _write_csv(out_dir / "poc_detection_metrics.csv", detection_rows)
    _write_csv(out_dir / "poc_isolation_metrics.csv", isolation_rows)
    _write_csv(out_dir / "poc_cross_domain_metrics.csv", cross_domain_rows)
    _write_csv(out_dir / "poc_performance_metrics.csv", performance_rows)

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
        performance_rows=performance_rows,
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
        "performance_metric_count": len(performance_rows),
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


def _performance_metrics(model_dir: Path, dataset_dir: Path) -> List[Dict[str, object]]:
    model, model_config, feature_names = load_sdae_model(model_dir)
    model.eval()
    parameter_count = int(sum(parameter.numel() for parameter in model.parameters()))
    nonzero_parameter_count = int(
        sum(int(torch.count_nonzero(parameter.detach()).item()) for parameter in model.parameters())
    )
    compute_estimates = sdae_compute_estimates(model_config)
    training_performance = _training_performance(model_dir)
    inference_rows = _inference_benchmark_rows(model, model_config, feature_names, model_dir, dataset_dir)
    rows = [
        _performance_row(
            "model",
            "model_parameter_count",
            parameter_count,
            "parameters",
            "counted_from_loaded_model",
            "Counted from the loaded PyTorch model parameters.",
        ),
        _performance_row(
            "model",
            "model_nonzero_parameter_count",
            nonzero_parameter_count,
            "parameters",
            "counted_from_loaded_model",
            "Counted non-zero values from the loaded PyTorch model parameters.",
        ),
        _performance_row(
            "model",
            "estimated_parameter_memory_fp32_mb",
            parameter_count * 4.0 / (1024.0 * 1024.0),
            "MB",
            "estimated_from_parameter_count",
            "Parameter count multiplied by 4 bytes for FP32 weights.",
        ),
        _performance_row(
            "compute_estimate",
            "estimated_forward_linear_macs_per_window",
            compute_estimates["estimated_forward_linear_macs_per_window"],
            "MACs per 100 ms window",
            "estimated_from_sdae_layer_sizes",
            "Sum of input_dim * output_dim for each SDAE Linear layer.",
        ),
        _performance_row(
            "compute_estimate",
            "estimated_forward_linear_flops_per_window",
            compute_estimates["estimated_forward_linear_flops_per_window"],
            "FLOPs per 100 ms window",
            "estimated_from_sdae_layer_sizes",
            "Linear-layer forward estimate. Multiply-adds count as 2 FLOPs plus bias additions.",
        ),
        _performance_row(
            "compute_estimate",
            "estimated_training_linear_flops_per_window",
            compute_estimates["estimated_training_linear_flops_per_window"],
            "FLOPs per training window",
            "estimated_from_sdae_layer_sizes",
            "Approximate training cost per window, using 3x the forward linear FLOP estimate.",
        ),
    ]
    rows.extend(_training_performance_rows(training_performance))
    rows.extend(inference_rows)
    return rows


def _training_performance(model_dir: Path) -> Dict[str, object]:
    metrics_path = model_dir / "metrics.json"
    if not metrics_path.exists():
        return {}
    with metrics_path.open("r", encoding="utf-8") as handle:
        metrics = json.load(handle)
    performance = metrics.get("performance")
    return performance if isinstance(performance, dict) else {}


def _training_performance_rows(performance: Dict[str, object]) -> List[Dict[str, object]]:
    if not performance:
        status = "not_available_legacy_model_artifact"
        return [
            _performance_row("training", "estimated_training_linear_flops_total", "", "FLOPs", status, "No training performance block was found in metrics.json."),
            _performance_row("training", "training_wall_time_s", "", "seconds", status, "No training performance block was found in metrics.json."),
            _performance_row("training", "training_cpu_time_s", "", "CPU seconds", status, "No training performance block was found in metrics.json."),
            _performance_row("training", "training_cpu_usage_percent_all_cores", "", "percent", status, "No training performance block was found in metrics.json."),
            _performance_row("training", "training_peak_ram_mb", "", "MB", status, "No training performance block was found in metrics.json."),
        ]
    status = "measured_during_train_sdae"
    return [
        _performance_row(
            "training",
            "estimated_training_linear_flops_total",
            performance.get("estimated_training_linear_flops_total", ""),
            "FLOPs",
            "estimated_from_sdae_layer_sizes",
            "Training windows multiplied by epochs and the approximate per-window training FLOP estimate.",
        ),
        _performance_row(
            "training",
            "train_window_epochs",
            performance.get("train_window_epochs", ""),
            "window-epochs",
            "counted_during_train_sdae",
            "Healthy training windows multiplied by configured epochs.",
        ),
        _performance_row(
            "training",
            "training_wall_time_s",
            performance.get("wall_time_s", ""),
            "seconds",
            status,
            "Wall-clock time measured during train_sdae.",
        ),
        _performance_row(
            "training",
            "training_cpu_time_s",
            performance.get("process_cpu_time_s", ""),
            "CPU seconds",
            status,
            "Process CPU time measured during train_sdae.",
        ),
        _performance_row(
            "training",
            "training_cpu_usage_percent_all_cores",
            performance.get("cpu_usage_percent_all_cores", ""),
            "percent of logical CPU capacity",
            status,
            "Process CPU time divided by wall time and logical CPU count during train_sdae.",
        ),
        _performance_row(
            "training",
            "training_peak_ram_mb",
            performance.get("peak_rss_mb", ""),
            "MB",
            status,
            "Peak resident memory sampled during train_sdae.",
        ),
    ]


def _inference_benchmark_rows(
    model: torch.nn.Module,
    model_config: Dict[str, object],
    feature_names: List[str],
    model_dir: Path,
    dataset_dir: Path,
) -> List[Dict[str, object]]:
    windows = pd.read_parquet(dataset_dir / "windows.parquet")
    if feature_names and list(windows.columns) != feature_names:
        windows = windows[feature_names]
    if windows.empty:
        return [
            _performance_row(
                "inference",
                "inference_benchmark_status",
                "not_available_no_windows",
                "status",
                "not_available_no_windows",
                "No windows were available for inference benchmarking.",
            )
        ]
    scaler = StandardFeatureScaler.load(model_dir / "scaler.joblib")
    benchmark_window_count = int(min(len(windows), 1024))
    values = windows.iloc[:benchmark_window_count].to_numpy(dtype=np.float32)
    minimum_repetitions = int(max(3, np.ceil(2048 / benchmark_window_count)))
    maximum_repetitions = 1000
    minimum_wall_s = 0.20
    with torch.no_grad():
        for _ in range(2):
            _run_inference_once(model, scaler, values)
        sampler = PerformanceSampler().start()
        repetitions = 0
        start_wall = time.perf_counter()
        while repetitions < maximum_repetitions:
            _run_inference_once(model, scaler, values)
            repetitions += 1
            if repetitions >= minimum_repetitions and (time.perf_counter() - start_wall) >= minimum_wall_s:
                break
        performance = sampler.stop()
    total_windows = int(benchmark_window_count * repetitions)
    wall_time_s = float(performance["wall_time_s"])
    cpu_time_s = float(performance["process_cpu_time_s"])
    throughput = total_windows / wall_time_s if wall_time_s > 0.0 else None
    compute_estimates = sdae_compute_estimates(model_config)
    forward_flops = float(compute_estimates["estimated_forward_linear_flops_per_window"])
    return [
        _performance_row(
            "inference",
            "inference_benchmark_window_count",
            benchmark_window_count,
            "windows",
            "measured_offline_cpu_benchmark",
            "Number of dataset windows used in each benchmark repetition.",
        ),
        _performance_row(
            "inference",
            "inference_benchmark_repetitions",
            repetitions,
            "repetitions",
            "measured_offline_cpu_benchmark",
            "Benchmark repeats used to reduce timer noise.",
        ),
        _performance_row(
            "inference",
            "inference_wall_time_ms_per_window",
            wall_time_s * 1000.0 / total_windows if total_windows else "",
            "ms per 100 ms window",
            "measured_offline_cpu_benchmark",
            "Wall-clock time for scaling, tensor conversion, SDAE forward pass, and reconstruction error.",
        ),
        _performance_row(
            "inference",
            "inference_cpu_time_ms_per_window",
            cpu_time_s * 1000.0 / total_windows if total_windows else "",
            "CPU ms per 100 ms window",
            "measured_offline_cpu_benchmark",
            "Process CPU time for scaling, tensor conversion, SDAE forward pass, and reconstruction error.",
        ),
        _performance_row(
            "inference",
            "inference_cpu_usage_percent_all_cores",
            performance.get("cpu_usage_percent_all_cores", ""),
            "percent of logical CPU capacity",
            "measured_offline_cpu_benchmark",
            "Process CPU time divided by wall time and logical CPU count during the inference benchmark.",
        ),
        _performance_row(
            "inference",
            "inference_peak_ram_mb",
            performance.get("peak_rss_mb", ""),
            "MB",
            "measured_offline_cpu_benchmark",
            "Peak resident memory sampled during the inference benchmark.",
        ),
        _performance_row(
            "inference",
            "inference_throughput_windows_per_second",
            throughput,
            "windows per second",
            "measured_offline_cpu_benchmark",
            "Total benchmarked windows divided by wall-clock seconds.",
        ),
        _performance_row(
            "inference",
            "estimated_forward_linear_flops_per_second",
            forward_flops * throughput if throughput is not None else "",
            "linear-layer FLOPs per second",
            "estimated_from_flops_and_measured_throughput",
            "Estimated forward linear FLOPs per window multiplied by measured benchmark throughput.",
        ),
        _performance_row(
            "inference",
            "logical_cpu_count",
            performance.get("logical_cpu_count", ""),
            "logical cores",
            "measured_offline_cpu_benchmark",
            "Logical CPU count reported by the operating system.",
        ),
    ]


def _run_inference_once(
    model: torch.nn.Module,
    scaler: StandardFeatureScaler,
    values: np.ndarray,
) -> None:
    scaled = scaler.transform(values)
    batch = torch.from_numpy(scaled)
    reconstruction, _latent = model(batch)
    errors = torch.mean((reconstruction - batch) ** 2, dim=1)
    float(errors.mean().item())


def _performance_row(
    category: str,
    metric: str,
    value: object,
    unit: str,
    status: str,
    method: str,
) -> Dict[str, object]:
    return {
        "category": category,
        "metric": metric,
        "value": value,
        "unit": unit,
        "status": status,
        "method": method,
    }


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
    performance_rows: List[Dict[str, object]],
) -> str:
    false_positive_rate = _metric_value(detection_rows, "overall", "false_positive_rate")
    true_fault_detection_rate = _metric_value(detection_rows, "overall", "true_fault_detection_rate")
    isolation_rate = _metric_value(isolation_rows, "overall", "true_fault_isolation_rate")
    cross_available = [row for row in cross_domain_rows if row["status"] == "available"]
    forward_flops = _performance_metric_value(performance_rows, "estimated_forward_linear_flops_per_window")
    training_flops = _performance_metric_value(performance_rows, "estimated_training_linear_flops_total")
    training_cpu = _performance_metric_value(performance_rows, "training_cpu_usage_percent_all_cores")
    training_ram = _performance_metric_value(performance_rows, "training_peak_ram_mb")
    inference_cpu = _performance_metric_value(performance_rows, "inference_cpu_usage_percent_all_cores")
    inference_ram = _performance_metric_value(performance_rows, "inference_peak_ram_mb")
    inference_latency = _performance_metric_value(performance_rows, "inference_wall_time_ms_per_window")
    source_type = str(dataset_manifest.get("source_type", "unknown"))
    evidence_note = (
        "This evaluation uses public CWRU bearing data with a reduced vibration-only profile. "
        "It is a software realism check, not final USV thesis evidence."
        if source_type == "external_cwru"
        else "This evaluation uses synthetic proof-of-concept data. "
        "It is a software smoke/evidence run, not final thesis evidence."
    )
    return "\n".join(
        [
            "# Proof-of-Concept Evaluation Summary",
            "",
            f"Created at: {datetime.now(timezone.utc).isoformat()}",
            "",
            evidence_note,
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
            f"- Estimated forward compute per 100 ms window: {_format_metric(forward_flops)} linear-layer FLOPs",
            f"- Estimated training compute: {_format_metric(training_flops)} linear-layer FLOPs",
            f"- Training CPU/RAM: {_format_metric(training_cpu)}% CPU, {_format_metric(training_ram)} MB peak RAM",
            f"- Inference CPU/RAM: {_format_metric(inference_cpu)}% CPU, {_format_metric(inference_ram)} MB peak RAM",
            f"- Inference latency: {_format_metric(inference_latency)} ms per 100 ms window",
            "",
            "## Acceptance Notes",
            "",
            "- SDAE anomaly decisions are computed from the saved reconstruction threshold.",
            "- Known/novel decisions use squared Mahalanobis distance against Ledoit-Wolf dictionary entries.",
            "- Cross-domain metrics are marked `not_available` when the dataset does not contain B1-B4 known fault anomaly windows.",
            "- CPU and RAM metrics are measured from the current process during training and offline inference benchmark runs.",
            "- FLOP counts are linear-layer estimates; activations, optimizer bookkeeping, data loading, and Python overhead are excluded.",
            "- Power measurements are not measured in this offline POC command.",
            "",
            "## Report Files",
            "",
            "- `poc_detection_metrics.csv`: detection reliability metrics",
            "- `poc_isolation_metrics.csv`: isolation metrics",
            "- `poc_cross_domain_metrics.csv`: B1-B4 transfer rows",
            "- `poc_performance_metrics.csv`: FLOP estimates plus CPU/RAM usage for training and inference",
            "- `poc_window_decisions.csv`: per-window anomaly and dictionary decisions",
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


def _performance_metric_value(rows: List[Dict[str, object]], metric: str) -> object:
    for row in rows:
        if row.get("metric") == metric:
            return row.get("value")
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
