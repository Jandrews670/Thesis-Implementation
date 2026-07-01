from __future__ import annotations

import argparse
import io
import re
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from usv_faults.clustering.fault_dictionary import load_fault_dictionary
from usv_faults.clustering.latent import infer_windows
from usv_faults.evaluation.reports import (
    _annotate_metric_warmup,
    _decisions_for_frame,
    _event_decisions_for_frame,
)
from usv_faults.preprocessing.windowing import feature_column_names


DEFAULT_ARCHIVE = Path("data/external/femto/_download/femto_nested/Training_set.zip")
DEFAULT_MODEL = Path("artifacts/models/run_public_femto_sdae")
DEFAULT_DICTIONARY = Path("artifacts/dictionaries/dict_public_femto_empirical")
DEFAULT_OUT = Path("runs/reports/public_femto_degradation_onset")
DEFAULT_BEARINGS = ["Bearing1_1", "Bearing1_2"]
DEFAULT_SNAPSHOT_INTERVAL_S = 10.0
FEATURE_DIM = 5120
SAMPLES_PER_AXIS = 2560


def main() -> int:
    args = _parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    dictionary = load_fault_dictionary(args.dictionary)

    all_rows: List[pd.DataFrame] = []
    summary_rows: List[Dict[str, object]] = []
    for bearing in args.bearing:
        windows, record_indices, members = _load_bearing_windows(
            args.archive,
            bearing,
            horizontal_column=args.horizontal_column,
            vertical_column=args.vertical_column,
        )
        inference = infer_windows(args.model, windows)
        frame = _inference_frame(
            bearing=bearing,
            record_indices=record_indices,
            members=members,
            inference=inference,
            snapshot_interval_s=args.snapshot_interval_s,
        )
        frame = _annotate_metric_warmup(frame, 0)
        decisions = _decisions_for_frame(frame, inference.latent_columns, dictionary)
        decisions = _restore_progression_columns(decisions, frame)
        events = _event_decisions_for_frame(decisions, dictionary)
        merged = _merged_rows(decisions, events)
        all_rows.append(merged)
        summary_rows.append(
            _summary_row(
                bearing=bearing,
                merged=merged,
                threshold=float(inference.threshold),
                snapshot_interval_s=args.snapshot_interval_s,
                stable_coverage_threshold=args.stable_coverage_threshold,
            )
        )

    windows_out = pd.concat(all_rows, ignore_index=True)
    summary = pd.DataFrame(summary_rows)
    windows_out.to_csv(args.out / "femto_degradation_onset_windows.csv", index=False)
    summary.to_csv(args.out / "femto_degradation_onset_summary.csv", index=False)
    (args.out / "femto_degradation_onset_summary.md").write_text(
        _summary_markdown(summary, args),
        encoding="utf-8",
    )

    print(f"Wrote FEMTO degradation onset analysis to {args.out}")
    print(summary.to_string(index=False))
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze FEMTO degradation onset from full learning-set snapshots.")
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--dictionary", type=Path, default=DEFAULT_DICTIONARY)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--bearing", action="append", default=None)
    parser.add_argument("--snapshot-interval-s", type=float, default=DEFAULT_SNAPSHOT_INTERVAL_S)
    parser.add_argument("--stable-coverage-threshold", type=float, default=0.80)
    parser.add_argument("--horizontal-column", type=int, default=4)
    parser.add_argument("--vertical-column", type=int, default=5)
    parsed = parser.parse_args()
    if parsed.bearing is None:
        parsed.bearing = DEFAULT_BEARINGS
    if parsed.snapshot_interval_s <= 0.0:
        raise ValueError("--snapshot-interval-s must be positive")
    if not 0.0 < parsed.stable_coverage_threshold <= 1.0:
        raise ValueError("--stable-coverage-threshold must be in (0, 1]")
    return parsed


def _load_bearing_windows(
    archive: Path,
    bearing: str,
    horizontal_column: int,
    vertical_column: int,
) -> tuple[pd.DataFrame, List[int], List[str]]:
    with zipfile.ZipFile(archive) as zf:
        members = sorted(
            [
                name
                for name in zf.namelist()
                if name.startswith(f"Learning_set/{bearing}/acc_") and name.endswith(".csv")
            ],
            key=_record_index,
        )
        if not members:
            raise ValueError(f"no FEMTO snapshots found for {bearing!r} in {archive}")
        features: List[np.ndarray] = []
        record_indices: List[int] = []
        for member in members:
            with zf.open(member) as handle:
                data = handle.read()
            values = np.loadtxt(
                io.BytesIO(data),
                delimiter=",",
                usecols=(horizontal_column, vertical_column),
                dtype=np.float32,
            )
            if values.ndim != 2 or values.shape[0] != SAMPLES_PER_AXIS or values.shape[1] != 2:
                raise ValueError(f"{member} has shape {values.shape}; expected ({SAMPLES_PER_AXIS}, 2)")
            features.append(np.concatenate([values[:, 0], values[:, 1]]).astype(np.float32, copy=False))
            record_indices.append(_record_index(member))
    return pd.DataFrame(features, columns=feature_column_names(FEATURE_DIM)), record_indices, members


def _record_index(path: str) -> int:
    match = re.search(r"acc_(\d+)\.csv$", path)
    if not match:
        raise ValueError(f"could not parse FEMTO record index from {path!r}")
    return int(match.group(1))


def _inference_frame(
    bearing: str,
    record_indices: List[int],
    members: List[str],
    inference,
    snapshot_interval_s: float,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    total = len(record_indices)
    for position, (record_index, member) in enumerate(zip(record_indices, members)):
        elapsed_s = float(position * snapshot_interval_s)
        row = {
            "trial_id": f"femto_{bearing}_full_life",
            "bearing": bearing,
            "record_index": int(record_index),
            "archive_member": member,
            "snapshot_position": int(position),
            "snapshots_to_failure": int(total - position - 1),
            "time_to_failure_s": float((total - position - 1) * snapshot_interval_s),
            "window_start_s": elapsed_s,
            "window_end_s": elapsed_s + snapshot_interval_s,
            "baseline_id": 0,
            "baseline_name": "femto_condition_1_full_life",
            "fault_label": "unknown_degradation_progression",
            "is_fault": False,
            "reconstruction_error": float(inference.reconstruction_errors[position]),
            "is_anomaly": bool(inference.is_anomaly[position]),
        }
        for column_index, latent_column in enumerate(inference.latent_columns):
            row[latent_column] = float(inference.latents[position, column_index])
        rows.append(row)
    return pd.DataFrame(rows)


def _merged_rows(decisions: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    event_columns = [
        "row_index",
        "event_decision",
        "event_fault_label",
        "event_confidence",
        "event_window_count",
        "event_anomaly_count",
        "event_anomaly_fraction",
        "event_known_vote_count",
        "event_top_known_vote_count",
        "event_novel_vote_count",
        "event_novel_family_vote_count",
        "event_known_vote_fraction",
        "event_novel_vote_fraction",
        "event_noise_vote_count",
        "event_insufficient_support_vote_count",
    ]
    event_subset = events[event_columns].copy()
    merged = decisions.merge(event_subset, on="row_index", how="left")
    return merged.drop(columns=[column for column in merged.columns if column.startswith("latent_")])


def _restore_progression_columns(decisions: pd.DataFrame, frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "bearing",
        "record_index",
        "archive_member",
        "snapshot_position",
        "snapshots_to_failure",
        "time_to_failure_s",
    ]
    extras = frame[columns].copy()
    extras.insert(0, "row_index", frame.index.astype(int))
    return decisions.merge(extras, on="row_index", how="left")


def _summary_row(
    bearing: str,
    merged: pd.DataFrame,
    threshold: float,
    snapshot_interval_s: float,
    stable_coverage_threshold: float,
) -> Dict[str, object]:
    anomaly = merged["is_anomaly"].astype(bool).to_numpy()
    event_alarm = merged["event_decision"].astype(str).isin({"known", "novel"}).to_numpy()
    event_known = (
        merged["event_decision"].astype(str).eq("known")
        & merged["event_fault_label"].astype(str).eq("bearing_degradation")
    ).to_numpy()
    onset_3_of_5 = _rolling_onset(anomaly, window=5, min_votes=3, min_fraction=0.60)
    onset_5_of_10 = _rolling_onset(anomaly, window=10, min_votes=5, min_fraction=0.50)
    first_anomaly = _first_true(anomaly)
    first_event_alarm = _first_true(event_alarm)
    first_known = _first_true(event_known)
    stable_anomaly = _stable_onset(anomaly, stable_coverage_threshold)
    stable_event_alarm = _stable_onset(event_alarm, stable_coverage_threshold)
    stable_known = _stable_onset(event_known, stable_coverage_threshold)
    return {
        "bearing": bearing,
        "snapshot_interval_s": snapshot_interval_s,
        "snapshot_count": int(len(merged)),
        "first_record_index": int(merged["record_index"].iloc[0]),
        "last_record_index": int(merged["record_index"].iloc[-1]),
        "reconstruction_threshold": threshold,
        "mean_reconstruction_error": float(merged["reconstruction_error"].mean()),
        "max_reconstruction_error": float(merged["reconstruction_error"].max()),
        "anomaly_fraction": float(np.mean(anomaly)),
        "final_20pct_anomaly_fraction": _tail_fraction(anomaly, 0.20),
        "stable_coverage_threshold": stable_coverage_threshold,
        **_onset_fields("first_anomaly", merged, first_anomaly, snapshot_interval_s),
        **_onset_fields("persistent_3_of_5", merged, onset_3_of_5, snapshot_interval_s),
        **_onset_fields("persistent_5_of_10", merged, onset_5_of_10, snapshot_interval_s),
        **_onset_fields("event_alarm", merged, first_event_alarm, snapshot_interval_s),
        **_onset_fields("known_degradation_event", merged, first_known, snapshot_interval_s),
        **_onset_fields("stable_anomaly", merged, stable_anomaly, snapshot_interval_s),
        **_onset_fields("stable_event_alarm", merged, stable_event_alarm, snapshot_interval_s),
        **_onset_fields("stable_known_degradation_event", merged, stable_known, snapshot_interval_s),
        "latency_event_from_first_anomaly_snapshots": _delta(first_event_alarm, first_anomaly),
        "latency_event_from_first_anomaly_s": _delta_s(first_event_alarm, first_anomaly, snapshot_interval_s),
        "latency_known_from_persistent_3_of_5_snapshots": _delta(first_known, onset_3_of_5),
        "latency_known_from_persistent_3_of_5_s": _delta_s(first_known, onset_3_of_5, snapshot_interval_s),
        "latency_stable_known_from_stable_anomaly_snapshots": _delta(stable_known, stable_anomaly),
        "latency_stable_known_from_stable_anomaly_s": _delta_s(stable_known, stable_anomaly, snapshot_interval_s),
        "event_alarm_coverage_after_onset": _coverage_after(event_alarm, first_event_alarm),
        "known_degradation_coverage_after_onset": _coverage_after(event_known, first_known),
        "stable_event_alarm_coverage_after_onset": _coverage_after(event_alarm, stable_event_alarm),
        "stable_known_degradation_coverage_after_onset": _coverage_after(event_known, stable_known),
        "max_event_alarm_dropout_after_onset_snapshots": _max_dropout_after(event_alarm, first_event_alarm),
        "max_known_dropout_after_onset_snapshots": _max_dropout_after(event_known, first_known),
        "max_stable_event_alarm_dropout_after_onset_snapshots": _max_dropout_after(event_alarm, stable_event_alarm),
        "max_stable_known_dropout_after_onset_snapshots": _max_dropout_after(event_known, stable_known),
    }


def _rolling_onset(flags: np.ndarray, window: int, min_votes: int, min_fraction: float) -> Optional[int]:
    for index in range(len(flags)):
        start = max(0, index - window + 1)
        history = flags[start : index + 1]
        votes = int(np.sum(history))
        if votes >= min_votes and votes / len(history) >= min_fraction:
            return int(index)
    return None


def _first_true(flags: np.ndarray) -> Optional[int]:
    indices = np.flatnonzero(flags)
    return int(indices[0]) if len(indices) else None


def _stable_onset(flags: np.ndarray, coverage_threshold: float) -> Optional[int]:
    for index, value in enumerate(flags):
        if value and float(np.mean(flags[index:])) >= coverage_threshold:
            return int(index)
    return None


def _tail_fraction(flags: np.ndarray, fraction: float) -> float:
    start = int(np.floor(len(flags) * (1.0 - fraction)))
    return float(np.mean(flags[start:])) if len(flags[start:]) else float("nan")


def _onset_fields(prefix: str, frame: pd.DataFrame, position: Optional[int], snapshot_interval_s: float) -> Dict[str, object]:
    if position is None:
        return {
            f"{prefix}_snapshot_position": "",
            f"{prefix}_record_index": "",
            f"{prefix}_elapsed_s": "",
            f"{prefix}_time_to_failure_s": "",
        }
    row = frame.iloc[position]
    return {
        f"{prefix}_snapshot_position": int(position),
        f"{prefix}_record_index": int(row["record_index"]),
        f"{prefix}_elapsed_s": float(position * snapshot_interval_s),
        f"{prefix}_time_to_failure_s": float(row["time_to_failure_s"]),
    }


def _delta(later: Optional[int], earlier: Optional[int]) -> object:
    if later is None or earlier is None:
        return ""
    return int(later - earlier)


def _delta_s(later: Optional[int], earlier: Optional[int], snapshot_interval_s: float) -> object:
    delta = _delta(later, earlier)
    return "" if delta == "" else float(delta * snapshot_interval_s)


def _coverage_after(flags: np.ndarray, onset: Optional[int]) -> object:
    if onset is None:
        return ""
    return float(np.mean(flags[onset:]))


def _max_dropout_after(flags: np.ndarray, onset: Optional[int]) -> object:
    if onset is None:
        return ""
    longest = 0
    current = 0
    for value in flags[onset:]:
        if value:
            current = 0
        else:
            current += 1
            longest = max(longest, current)
    return int(longest)


def _summary_markdown(summary: pd.DataFrame, args: argparse.Namespace) -> str:
    rows = [
        "# FEMTO Degradation Onset Analysis",
        "",
        "This analysis treats FEMTO/PRONOSTIA as a progressive degradation dataset rather than a clean fault-isolation dataset.",
        "",
        "Inputs:",
        "",
        f"- Archive: `{args.archive}`",
        f"- Model: `{args.model}`",
        f"- Dictionary: `{args.dictionary}`",
        f"- Snapshot interval assumption: `{args.snapshot_interval_s}` seconds per FEMTO CSV snapshot",
        f"- Stable onset coverage threshold: `{args.stable_coverage_threshold}`",
        "",
        "Definitions:",
        "",
        "- `first_anomaly`: first snapshot whose SDAE reconstruction error exceeds the saved threshold.",
        "- `persistent_3_of_5`: first snapshot where at least 3 of the latest 5 snapshots are anomalous.",
        "- `persistent_5_of_10`: first snapshot where at least 5 of the latest 10 snapshots are anomalous.",
        "- `event_alarm`: first rolling event decision of `known` or `novel`.",
        "- `known_degradation_event`: first rolling event decision of `known` with label `bearing_degradation`.",
        "- `stable_known_degradation_event`: first known degradation event where at least the configured fraction of all remaining snapshots are also known degradation events.",
        "",
        "## Summary",
        "",
        "| Bearing | Snapshots | Stable anomaly record | Stable known record | Stable known lead time to failure | Stable known latency from stable anomaly | Stable known coverage |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.to_dict(orient="records"):
        rows.append(
            "| {bearing} | {snapshot_count} | {stable_anomaly_record_index} | "
            "{stable_known_degradation_event_record_index} | {stable_known_degradation_event_time_to_failure_s} s | "
            "{latency_stable_known_from_stable_anomaly_s} s | {stable_known_degradation_coverage_after_onset:.3f} |".format(
                **_format_row(row)
            )
        )
    rows.extend(
        [
            "",
            "## Interpretation",
            "",
            "These are onset/lead-time statistics, not fault-type isolation scores. They are more appropriate for FEMTO because degradation ramps over time rather than starting as a clean induced class at a known timestamp.",
            "",
            "The `time_to_failure_s` values use the configured snapshot interval assumption. The CSV also records snapshot positions and record indices so the result remains auditable if a different wall-clock cadence is preferred.",
            "",
            "Detailed per-snapshot outputs are in `femto_degradation_onset_windows.csv`; summary fields are in `femto_degradation_onset_summary.csv`.",
            "",
        ]
    )
    return "\n".join(rows)


def _format_row(row: Dict[str, object]) -> Dict[str, object]:
    formatted = dict(row)
    for key, value in formatted.items():
        if value == "" or value is None:
            formatted[key] = "n/a"
        elif isinstance(value, float):
            formatted[key] = round(value, 3)
    return formatted


if __name__ == "__main__":
    raise SystemExit(main())
