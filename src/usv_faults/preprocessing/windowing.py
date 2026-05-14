from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from usv_faults.schemas import EventRow, TrialManifest


@dataclass
class WindowingConfig:
    window_ms: float
    stride_ms: float
    current_sample_rate_hz: int
    scalar_features: List[str]
    expected_input_dim: int


def feature_column_names(input_dim: int) -> List[str]:
    return [f"feature_{index:04d}" for index in range(input_dim)]


def build_windows_for_trial(
    telemetry: pd.DataFrame,
    manifest: TrialManifest,
    events: List[EventRow],
    config: WindowingConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    raw_rate_hz = manifest.sensor_config.sample_rate_hz
    window_samples = int(round(raw_rate_hz * config.window_ms / 1000.0))
    stride_samples = int(round(raw_rate_hz * config.stride_ms / 1000.0))
    if window_samples <= 0 or stride_samples <= 0:
        raise ValueError("window and stride must be positive")
    if raw_rate_hz % config.current_sample_rate_hz != 0:
        raise ValueError("current decimation requires an integer sample-rate ratio")
    current_decimation = raw_rate_hz // config.current_sample_rate_hz

    vibration_channels = manifest.sensor_config.all_vibration_channels
    current_channels = manifest.sensor_config.current_channels
    scalar_channels = manifest.sensor_config.scalar_channels
    required = ["timestamp_s"] + vibration_channels + current_channels + scalar_channels
    missing = [channel for channel in required if channel not in telemetry.columns]
    if missing:
        raise ValueError(f"telemetry missing required channels: {missing}")

    fault_intervals = _fault_intervals_from_events(events, manifest)
    features: List[np.ndarray] = []
    labels: List[Dict[str, object]] = []

    max_start = len(telemetry) - window_samples
    for start_index in range(0, max_start + 1, stride_samples):
        end_index = start_index + window_samples
        window = telemetry.iloc[start_index:end_index]
        start_s = float(window["timestamp_s"].iloc[0])
        end_s = float(window["timestamp_s"].iloc[-1]) + (1.0 / raw_rate_hz)
        vector = _feature_vector(
            window,
            vibration_channels,
            current_channels,
            scalar_channels,
            config.scalar_features,
            current_decimation,
        )
        if len(vector) != config.expected_input_dim:
            raise ValueError(
                f"window feature dimension {len(vector)} does not match "
                f"expected {config.expected_input_dim}"
            )
        fault_label = _fault_label_for_window(start_s, end_s, fault_intervals)
        features.append(vector)
        labels.append(
            {
                "trial_id": manifest.trial_id,
                "window_start_s": start_s,
                "window_end_s": end_s,
                "baseline_id": manifest.baseline.id,
                "baseline_name": manifest.baseline.name,
                "fault_label": fault_label or "none",
                "is_fault": bool(fault_label),
            }
        )

    windows = pd.DataFrame(features, columns=feature_column_names(config.expected_input_dim))
    labels_df = pd.DataFrame(labels)
    return windows, labels_df


def _feature_vector(
    window: pd.DataFrame,
    vibration_channels: List[str],
    current_channels: List[str],
    scalar_channels: List[str],
    scalar_features: List[str],
    current_decimation: int,
) -> np.ndarray:
    parts: List[np.ndarray] = []
    for channel in vibration_channels:
        parts.append(window[channel].to_numpy(dtype=np.float64))
    for channel in current_channels:
        parts.append(window[channel].to_numpy(dtype=np.float64)[::current_decimation])
    for channel in scalar_channels:
        values = window[channel].to_numpy(dtype=np.float64)
        for feature in scalar_features:
            if feature == "mean":
                parts.append(np.array([float(np.mean(values))], dtype=np.float64))
            elif feature == "variance":
                parts.append(np.array([float(np.var(values))], dtype=np.float64))
            elif feature == "peak_to_peak":
                parts.append(np.array([float(np.ptp(values))], dtype=np.float64))
            else:
                raise ValueError(f"unknown scalar feature: {feature}")
    return np.concatenate(parts)


def _fault_intervals_from_events(
    events: List[EventRow],
    manifest: TrialManifest,
) -> List[Tuple[float, float, str]]:
    intervals: List[Tuple[float, float, str]] = []
    active_start: Optional[float] = None
    active_label: Optional[str] = None
    for event in sorted(events, key=lambda item: item.timestamp_s):
        if event.event_type == "fault_start":
            active_start = event.timestamp_s
            active_label = event.value or manifest.fault.type
        elif event.event_type == "fault_end" and active_start is not None:
            intervals.append((active_start, event.timestamp_s, active_label or manifest.fault.type))
            active_start = None
            active_label = None
    if active_start is not None and manifest.fault.end_time_s is not None:
        intervals.append((active_start, manifest.fault.end_time_s, active_label or manifest.fault.type))
    return intervals


def _fault_label_for_window(
    start_s: float,
    end_s: float,
    fault_intervals: List[Tuple[float, float, str]],
) -> Optional[str]:
    for fault_start_s, fault_end_s, label in fault_intervals:
        if start_s < fault_end_s and end_s > fault_start_s:
            return label
    return None
