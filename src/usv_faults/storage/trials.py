from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import List

import pandas as pd

from usv_faults.config import read_yaml, write_yaml
from usv_faults.schemas import EventRow, QualityReport, TrialManifest


REQUIRED_TRIAL_FILES = ("manifest.yaml", "telemetry.parquet", "events.csv", "notes.md")


def write_manifest(path: Path, manifest: TrialManifest) -> None:
    write_yaml(path, manifest.model_dump(mode="json", exclude_none=True))


def read_manifest(trial_dir: Path) -> TrialManifest:
    return TrialManifest.model_validate(read_yaml(trial_dir / "manifest.yaml"))


def write_events(path: Path, events: List[EventRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["timestamp_s", "event_type", "value", "notes"])
        writer.writeheader()
        for event in events:
            writer.writerow(event.model_dump(mode="json"))


def read_events(trial_dir: Path) -> List[EventRow]:
    events_path = trial_dir / "events.csv"
    with events_path.open("r", encoding="utf-8", newline="") as handle:
        rows = csv.DictReader(handle)
        return [
            EventRow(
                timestamp_s=float(row["timestamp_s"]),
                event_type=row["event_type"],
                value=row.get("value") or None,
                notes=row.get("notes") or None,
            )
            for row in rows
        ]


def write_quality_report(trial_dir: Path, report: QualityReport) -> None:
    with (trial_dir / "quality_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report.model_dump(mode="json"), handle, indent=2)


def quality_check_trial(trial_dir: Path) -> QualityReport:
    errors: List[str] = []
    warnings: List[str] = []
    channel_checks: dict = {}
    event_checks: dict = {}

    for filename in REQUIRED_TRIAL_FILES:
        if not (trial_dir / filename).exists():
            errors.append(f"missing required file: {filename}")

    if errors:
        report = QualityReport(
            trial_id=trial_dir.name,
            status="rejected",
            sample_count=0,
            duration_s=0.0,
            sample_rate_hz_estimate=0.0,
            channel_checks=channel_checks,
            event_checks=event_checks,
            warnings=warnings,
            errors=errors,
        )
        write_quality_report(trial_dir, report)
        return report

    manifest = read_manifest(trial_dir)
    events = read_events(trial_dir)
    telemetry = pd.read_parquet(trial_dir / "telemetry.parquet")

    expected_channels = (
        ["timestamp_s"]
        + manifest.sensor_config.all_vibration_channels
        + manifest.sensor_config.current_channels
        + manifest.sensor_config.scalar_channels
    )

    for channel in expected_channels:
        if channel not in telemetry.columns:
            errors.append(f"missing telemetry channel: {channel}")
            channel_checks[channel] = {"present": False}
            continue
        series = telemetry[channel]
        all_null = bool(series.isna().all())
        flatlined = bool(series.nunique(dropna=True) <= 1)
        channel_checks[channel] = {
            "present": True,
            "all_null": all_null,
            "flatlined": flatlined,
        }
        if all_null:
            errors.append(f"all-null telemetry channel: {channel}")
        expected_constant_channels = {"voltage", "water_temperature", "pwm_command"}
        if flatlined and channel not in expected_constant_channels:
            warnings.append(f"flatlined telemetry channel: {channel}")

    sample_count = int(len(telemetry))
    if sample_count < 2 or "timestamp_s" not in telemetry.columns:
        duration_s = 0.0
        sample_rate_hz_estimate = 0.0
        errors.append("not enough timestamped samples to estimate sample rate")
    else:
        timestamps = telemetry["timestamp_s"]
        monotonic = bool(timestamps.is_monotonic_increasing)
        if not monotonic:
            errors.append("timestamps are not monotonic increasing")
        duration_s = float(timestamps.iloc[-1] - timestamps.iloc[0])
        sample_rate_hz_estimate = float((sample_count - 1) / duration_s) if duration_s > 0 else 0.0
        expected_rate = manifest.sensor_config.sample_rate_hz
        rate_error = abs(sample_rate_hz_estimate - expected_rate) / expected_rate
        if rate_error > 0.01:
            warnings.append(
                f"sample rate estimate {sample_rate_hz_estimate:.2f} Hz differs from "
                f"expected {expected_rate} Hz by more than 1 percent"
            )

    event_types = {event.event_type for event in events}
    event_checks["event_types"] = sorted(event_types)
    event_checks["has_trial_start"] = "trial_start" in event_types
    event_checks["has_trial_end"] = "trial_end" in event_types
    if "trial_start" not in event_types:
        errors.append("missing trial_start event")
    if "trial_end" not in event_types:
        errors.append("missing trial_end event")
    if manifest.fault.induced:
        event_checks["has_fault_start"] = "fault_start" in event_types
        event_checks["has_fault_end"] = "fault_end" in event_types
        if "fault_start" not in event_types or "fault_end" not in event_types:
            errors.append("fault trial missing fault_start or fault_end event")

    status = "rejected" if errors else ("accepted_with_notes" if warnings else "accepted")
    report = QualityReport(
        trial_id=manifest.trial_id,
        status=status,
        sample_count=sample_count,
        duration_s=duration_s,
        sample_rate_hz_estimate=sample_rate_hz_estimate,
        channel_checks=channel_checks,
        event_checks=event_checks,
        warnings=warnings,
        errors=errors,
    )
    write_quality_report(trial_dir, report)
    return report
