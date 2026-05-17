from __future__ import annotations

import hashlib
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from usv_faults.config import read_yaml
from usv_faults.schemas import EventRow, TrialManifest
from usv_faults.storage.trials import REQUIRED_TRIAL_FILES, quality_check_trial, write_events, write_manifest


class CWRUBearingSource:
    """Attach selected Case Western Reserve University bearing .mat files."""

    def __init__(self, config: Dict[str, object], project_root: Path) -> None:
        self.config = config
        self.project_root = project_root

    @classmethod
    def from_config_path(cls, path: Path) -> "CWRUBearingSource":
        config_path = path.resolve()
        project_root = config_path.parent.parent if config_path.parent.name == "configs" else config_path.parent
        return cls(read_yaml(config_path), project_root)

    def attach(self, out_dir: Path) -> List[Path]:
        out_dir.mkdir(parents=True, exist_ok=True)
        created: List[Path] = []
        for trial in list(self.config.get("trials", [])):
            if not isinstance(trial, dict):
                raise ValueError("each CWRU trial entry must be a mapping")
            trial_dir = out_dir / str(trial["trial_id"])
            trial_dir.mkdir(parents=True, exist_ok=True)
            if all((trial_dir / filename).exists() for filename in REQUIRED_TRIAL_FILES):
                quality_check_trial(trial_dir)
                created.append(trial_dir)
                continue

            mat_path = self._ensure_mat_file(trial)
            telemetry, mat_variables = self._telemetry_from_mat(mat_path, trial)
            telemetry.to_parquet(trial_dir / "telemetry.parquet", index=False)

            manifest = self._manifest_for(trial, telemetry, mat_path, mat_variables)
            write_manifest(trial_dir / "manifest.yaml", manifest)
            duration_s = float(len(telemetry) / int(self.config.get("sampling", {}).get("raw_sample_rate_hz", 12000)))
            write_events(trial_dir / "events.csv", self._events_for(trial, duration_s))
            (trial_dir / "notes.md").write_text(self._notes_for(trial, mat_path), encoding="utf-8")
            quality_check_trial(trial_dir)
            created.append(trial_dir)
        return created

    def _ensure_mat_file(self, trial: Dict[str, object]) -> Path:
        file_name = str(trial["file_name"])
        configured_path = trial.get("path")
        if configured_path:
            mat_path = self._resolve_path(Path(str(configured_path)))
            if not mat_path.exists():
                raise FileNotFoundError(f"CWRU trial file not found: {mat_path}")
            return mat_path

        cache_dir = self._resolve_path(Path(str(self.config.get("cache_dir", "data/external/cwru"))))
        cache_dir.mkdir(parents=True, exist_ok=True)
        mat_path = cache_dir / file_name
        if mat_path.exists():
            self._check_md5(mat_path, trial)
            return mat_path

        url = str(trial.get("url") or "").strip()
        if not url:
            raise FileNotFoundError(f"{mat_path} does not exist and no download URL was configured")

        tmp_path = mat_path.with_suffix(mat_path.suffix + ".download")
        urllib.request.urlretrieve(url, tmp_path)
        tmp_path.replace(mat_path)
        self._check_md5(mat_path, trial)
        return mat_path

    def _telemetry_from_mat(self, mat_path: Path, trial: Dict[str, object]) -> tuple[pd.DataFrame, Dict[str, str]]:
        try:
            from scipy.io import loadmat
        except ImportError as exc:  # pragma: no cover - dependency is declared in pyproject.
            raise RuntimeError("scipy is required to read CWRU MATLAB files") from exc

        mat = loadmat(mat_path)
        sample_rate_hz = int(self.config.get("sampling", {}).get("raw_sample_rate_hz", 12000))
        if sample_rate_hz <= 0:
            raise ValueError("sampling.raw_sample_rate_hz must be positive")

        profile = self._channel_profile()
        data: Dict[str, np.ndarray] = {}
        variable_map: Dict[str, str] = {}
        for channel in profile.get("vibration_channels", []):
            variable_name = self._variable_for_channel(mat, trial, str(channel))
            data[str(channel)] = self._series_from_mat(mat, variable_name)
            variable_map[str(channel)] = variable_name
        for channel in profile.get("current_channels", []):
            variable_name = self._variable_for_channel(mat, trial, str(channel))
            data[str(channel)] = self._series_from_mat(mat, variable_name)
            variable_map[str(channel)] = variable_name

        if not data:
            raise ValueError("CWRU channel profile must include at least one signal channel")

        length = min(len(values) for values in data.values())
        start_sample = int(trial.get("start_sample", 0))
        if start_sample < 0 or start_sample >= length:
            raise ValueError(f"invalid start_sample {start_sample} for {mat_path}")
        duration_s = trial.get("duration_s", self.config.get("default_duration_s"))
        if duration_s is not None:
            sample_count = int(round(float(duration_s) * sample_rate_hz))
        else:
            sample_count = length - start_sample
        end_sample = min(length, start_sample + sample_count)
        if end_sample <= start_sample:
            raise ValueError(f"trial {trial['trial_id']} contains no samples after trimming")

        trimmed = {
            channel: values[start_sample:end_sample].astype(np.float64, copy=False)
            for channel, values in data.items()
        }
        timestamps = np.arange(end_sample - start_sample, dtype=np.float64) / sample_rate_hz
        frame = pd.DataFrame({"timestamp_s": timestamps, **trimmed})
        return frame, variable_map

    def _variable_for_channel(self, mat: Dict[str, object], trial: Dict[str, object], channel: str) -> str:
        mappings = trial.get("mat_variables", {})
        if isinstance(mappings, dict) and channel in mappings:
            variable_name = str(mappings[channel])
            if variable_name not in mat:
                raise ValueError(f"MAT variable {variable_name!r} was not found")
            return variable_name
        if trial.get("mat_variable"):
            variable_name = str(trial["mat_variable"])
            if variable_name not in mat:
                raise ValueError(f"MAT variable {variable_name!r} was not found")
            return variable_name
        return self._infer_drive_end_variable(mat)

    def _infer_drive_end_variable(self, mat: Dict[str, object]) -> str:
        candidates = [key for key in mat if key.endswith("_DE_time") or "DE_time" in key]
        if not candidates:
            candidates = [
                key
                for key, value in mat.items()
                if not key.startswith("__") and np.asarray(value).size > 1 and np.asarray(value).ndim <= 2
            ]
        if not candidates:
            raise ValueError("could not infer a usable CWRU signal variable from MAT file")
        return sorted(candidates)[0]

    def _series_from_mat(self, mat: Dict[str, object], variable_name: str) -> np.ndarray:
        values = np.asarray(mat[variable_name], dtype=np.float64).squeeze()
        if values.ndim != 1:
            raise ValueError(f"MAT variable {variable_name!r} is not a 1-D signal")
        if values.size < 2:
            raise ValueError(f"MAT variable {variable_name!r} does not contain enough samples")
        return values

    def _manifest_for(
        self,
        trial: Dict[str, object],
        telemetry: pd.DataFrame,
        mat_path: Path,
        mat_variables: Dict[str, str],
    ) -> TrialManifest:
        profile = self._channel_profile()
        sample_rate_hz = int(self.config.get("sampling", {}).get("raw_sample_rate_hz", 12000))
        duration_s = float(len(telemetry) / sample_rate_hz)
        fault_label = str(trial.get("fault_label", "none"))
        induced = fault_label != "none" and bool(trial.get("fault_induced", True))
        source_type = str(self.config.get("source_type", "external_cwru"))
        return TrialManifest(
            trial_id=str(trial["trial_id"]),
            created_at=datetime.now(timezone.utc).isoformat(),
            operator="external_cwru_adapter",
            hardware={
                "motor": "CWRU bearing test rig",
                "bearing_dataset": "Case Western Reserve University Bearing Data Center",
            },
            sensor_config={
                "sample_rate_hz": sample_rate_hz,
                "vibration_channels": list(profile.get("vibration_channels", [])),
                "current_channels": list(profile.get("current_channels", [])),
                "scalar_channels": list(profile.get("scalar_channels", [])),
                "adc_resolution_bits": None,
            },
            baseline={
                "id": int(trial.get("baseline_id", 0)),
                "name": str(trial.get("baseline_name", "cwru_nominal")),
                "rpm": trial.get("rpm"),
                "load_hp": trial.get("load_hp"),
            },
            fault={
                "induced": induced,
                "type": fault_label,
                "start_time_s": 0.0 if induced else None,
                "end_time_s": duration_s if induced else None,
            },
            collection={
                "duration_s": duration_s,
                "source_type": source_type,
                "source_version": str(self.config.get("schema_version", "0.1.0")),
                "source_url": str(trial.get("url") or self.config.get("source_url", "")),
                "source_file": mat_path.name,
                "source_md5": self._md5(mat_path),
                "source_dataset": "CWRU Bearing Data Center",
                "channel_profile": profile,
                "mat_variables": mat_variables,
                "start_sample": int(trial.get("start_sample", 0)),
                "reduced_profile_reason": (
                    "CWRU is vibration-only in this adapter, so it uses a separate "
                    "reduced channel profile instead of padding to the 2109-D USV schema."
                ),
            },
            notes=(
                f"CWRU public bearing trial from {mat_path.name}; fault={fault_label}; "
                "reduced vibration-only profile."
            ),
        )

    def _events_for(self, trial: Dict[str, object], duration_s: float) -> List[EventRow]:
        fault_label = str(trial.get("fault_label", "none"))
        induced = fault_label != "none" and bool(trial.get("fault_induced", True))
        events = [EventRow(timestamp_s=0.0, event_type="trial_start", value=None, notes=None)]
        if induced:
            events.append(EventRow(timestamp_s=0.0, event_type="fault_start", value=fault_label, notes=None))
            events.append(EventRow(timestamp_s=duration_s, event_type="fault_end", value=fault_label, notes=None))
        events.append(EventRow(timestamp_s=duration_s, event_type="trial_end", value=None, notes=None))
        return events

    def _notes_for(self, trial: Dict[str, object], mat_path: Path) -> str:
        return "\n".join(
            [
                f"# {trial['trial_id']}",
                "",
                "Public CWRU bearing dataset trial converted to the canonical raw-trial folder layout.",
                "",
                f"- Source file: `{mat_path.name}`",
                f"- Source URL: `{trial.get('url') or self.config.get('source_url', '')}`",
                "- Channel profile: reduced vibration-only profile; missing USV current channels are not padded.",
                "",
            ]
        )

    def _channel_profile(self) -> Dict[str, object]:
        profile = self.config.get("channel_profile", {})
        if not isinstance(profile, dict):
            raise ValueError("channel_profile must be a mapping")
        return profile

    def _resolve_path(self, path: Path) -> Path:
        return path if path.is_absolute() else self.project_root / path

    def _check_md5(self, path: Path, trial: Dict[str, object]) -> None:
        expected = str(trial.get("md5") or "").strip().lower()
        if expected and self._md5(path).lower() != expected:
            raise ValueError(f"MD5 mismatch for {path}: expected {expected}, got {self._md5(path)}")

    def _md5(self, path: Path) -> str:
        digest = hashlib.md5()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
