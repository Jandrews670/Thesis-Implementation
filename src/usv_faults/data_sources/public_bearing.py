from __future__ import annotations

import hashlib
import re
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from usv_faults.config import read_yaml
from usv_faults.schemas import EventRow, TrialManifest
from usv_faults.storage.trials import REQUIRED_TRIAL_FILES, quality_check_trial, write_events, write_manifest


class PublicBearingSource:
    """Attach public bearing datasets from local delimited files or MATLAB files.

    The adapter is intentionally config-driven because public bearing datasets use many small
    format variations. It supports the current CWRU-like raw-trial contract without baking a
    fragile parser for each dataset release.
    """

    def __init__(self, config: Dict[str, object], project_root: Path, source_key: str) -> None:
        self.config = config
        self.project_root = project_root
        self.source_key = source_key

    @classmethod
    def from_config_path(cls, path: Path, source_key: str = "public_bearing") -> "PublicBearingSource":
        config_path = path.resolve()
        project_root = config_path.parent.parent if config_path.parent.name == "configs" else config_path.parent
        return cls(read_yaml(config_path), project_root, source_key)

    def attach(self, out_dir: Path) -> List[Path]:
        out_dir.mkdir(parents=True, exist_ok=True)
        created: List[Path] = []
        for trial in list(self.config.get("trials", [])):
            if not isinstance(trial, dict):
                raise ValueError("each public bearing trial entry must be a mapping")
            trial_dir = out_dir / str(trial["trial_id"])
            trial_dir.mkdir(parents=True, exist_ok=True)
            if all((trial_dir / filename).exists() for filename in REQUIRED_TRIAL_FILES):
                quality_check_trial(trial_dir)
                created.append(trial_dir)
                continue

            telemetry, source_files, source_variables = self._telemetry_for_trial(trial)
            telemetry.to_parquet(trial_dir / "telemetry.parquet", index=False)

            manifest = self._manifest_for(trial, telemetry, source_files, source_variables)
            write_manifest(trial_dir / "manifest.yaml", manifest)
            duration_s = float(telemetry["timestamp_s"].iloc[-1] + (1.0 / self._sample_rate_hz(trial)))
            write_events(trial_dir / "events.csv", self._events_for(trial, duration_s))
            (trial_dir / "notes.md").write_text(self._notes_for(trial, source_files), encoding="utf-8")
            quality_check_trial(trial_dir)
            created.append(trial_dir)
        return created

    def _telemetry_for_trial(
        self,
        trial: Dict[str, object],
    ) -> Tuple[pd.DataFrame, List[Path], Dict[str, str]]:
        sample_rate_hz = self._sample_rate_hz(trial)
        records = self._expanded_records(trial)
        if not records:
            records = [{}]

        channel_parts: Dict[str, List[np.ndarray]] = {channel: [] for channel in self._signal_channels()}
        source_files: List[Path] = []
        source_variables: Dict[str, str] = {}
        for record in records:
            if not isinstance(record, dict):
                raise ValueError("each record entry must be a mapping")
            spec = dict(trial)
            spec.update(record)
            path = self._ensure_source_file(spec)
            source_files.append(path)
            values, variables = self._read_source_file(path, spec)
            source_variables.update(variables)
            for channel in channel_parts:
                if channel not in values:
                    raise ValueError(f"{path} did not provide configured channel {channel!r}")
                channel_parts[channel].append(values[channel])

        concatenated = {
            channel: np.concatenate(parts).astype(np.float64, copy=False)
            for channel, parts in channel_parts.items()
        }
        length = min(len(values) for values in concatenated.values())
        start_sample = int(trial.get("start_sample", 0))
        if start_sample < 0 or start_sample >= length:
            raise ValueError(f"invalid start_sample {start_sample} for trial {trial['trial_id']}")
        duration_s = trial.get("duration_s", self.config.get("default_duration_s"))
        if duration_s is None:
            sample_count = length - start_sample
        else:
            sample_count = int(round(float(duration_s) * sample_rate_hz))
        end_sample = min(length, start_sample + sample_count)
        if end_sample <= start_sample:
            raise ValueError(f"trial {trial['trial_id']} contains no samples after trimming")

        trimmed = {
            channel: values[start_sample:end_sample]
            for channel, values in concatenated.items()
        }
        timestamps = np.arange(end_sample - start_sample, dtype=np.float64) / sample_rate_hz
        return pd.DataFrame({"timestamp_s": timestamps, **trimmed}), source_files, source_variables

    def _expanded_records(self, trial: Dict[str, object]) -> List[Dict[str, object]]:
        records = [dict(record) for record in list(trial.get("records", []))]
        for record_range in list(trial.get("record_ranges", [])):
            if not isinstance(record_range, dict):
                raise ValueError("each record_ranges entry must be a mapping")
            template = str(record_range["path_template"])
            start = int(record_range["start"])
            end = int(record_range["end"])
            step = int(record_range.get("step", 1))
            if step <= 0 or end < start:
                raise ValueError(f"invalid record range for trial {trial['trial_id']}: {record_range}")
            extras = {
                key: value
                for key, value in record_range.items()
                if key not in {"path_template", "start", "end", "step"}
            }
            for index in range(start, end + 1, step):
                record = dict(extras)
                record["path"] = template.format(index=index)
                records.append(record)
        return records

    def _read_source_file(
        self,
        path: Path,
        spec: Dict[str, object],
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, str]]:
        fmt = str(spec.get("format", self.config.get("format", path.suffix.lstrip(".") or "delimited"))).lower()
        if fmt in {"csv", "txt", "tsv", "delimited", "ascii"}:
            return self._read_delimited(path, spec)
        if fmt in {"mat", "matlab"}:
            return self._read_mat(path, spec)
        raise ValueError(f"unsupported public bearing file format {fmt!r} for {path}")

    def _read_delimited(
        self,
        path: Path,
        spec: Dict[str, object],
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, str]]:
        delimiter = str(spec.get("delimiter", self.config.get("delimiter", "auto")))
        sep = r"[\s,;]+" if delimiter == "auto" else delimiter
        has_header = bool(spec.get("has_header", self.config.get("has_header", False)))
        decimal = str(spec.get("decimal", self.config.get("decimal", ".")))
        comment = spec.get("comment", self.config.get("comment"))
        frame = pd.read_csv(
            path,
            sep=sep,
            engine="python",
            header=0 if has_header else None,
            comment=str(comment) if comment is not None else None,
            decimal=decimal,
        )
        frame = frame.apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all")
        columns = self._columns_for(spec, frame)
        values: Dict[str, np.ndarray] = {}
        variables: Dict[str, str] = {}
        for channel, column in columns.items():
            series = frame[column]
            values[channel] = series.to_numpy(dtype=np.float64)
            variables[channel] = str(column)
        return values, variables

    def _read_mat(
        self,
        path: Path,
        spec: Dict[str, object],
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, str]]:
        try:
            from scipy.io import loadmat
        except ImportError as exc:  # pragma: no cover - dependency is declared in pyproject.
            raise RuntimeError("scipy is required to read MATLAB public bearing files") from exc

        mat = loadmat(path, squeeze_me=True, struct_as_record=False)
        values: Dict[str, np.ndarray] = {}
        variables: Dict[str, str] = {}
        columns = dict(spec.get("columns", self.config.get("columns", {})) or {})
        mat_variables = dict(spec.get("mat_variables", self.config.get("mat_variables", {})) or {})
        shared_variable = spec.get("mat_variable", self.config.get("mat_variable"))

        for channel in self._signal_channels():
            variable_path = str(mat_variables.get(channel, shared_variable or ""))
            if variable_path:
                raw = self._resolve_mat_path(mat, variable_path)
                variable_label = variable_path
            else:
                variable_label, raw = self._infer_mat_signal(mat, channel)
            array = np.asarray(raw, dtype=np.float64).squeeze()
            if array.ndim == 0:
                raise ValueError(f"MAT variable {variable_label!r} for {channel!r} is scalar")
            column = columns.get(channel)
            if array.ndim > 1:
                if column is None:
                    column = 0
                array = array[:, int(column)]
            values[channel] = np.asarray(array, dtype=np.float64).reshape(-1)
            variables[channel] = variable_label if column is None else f"{variable_label}[{column}]"
        return values, variables

    def _columns_for(self, spec: Dict[str, object], frame: pd.DataFrame) -> Dict[str, object]:
        configured = dict(spec.get("columns", self.config.get("columns", {})) or {})
        if configured:
            return {
                channel: self._normalise_column(column, frame)
                for channel, column in configured.items()
                if channel in set(self._signal_channels())
            }
        channels = self._signal_channels()
        if len(frame.columns) < len(channels):
            raise ValueError(
                f"file has {len(frame.columns)} numeric columns but {len(channels)} signal channels are configured"
            )
        return {channel: frame.columns[index] for index, channel in enumerate(channels)}

    def _normalise_column(self, column: object, frame: pd.DataFrame) -> object:
        if isinstance(column, int):
            return frame.columns[column]
        if isinstance(column, str) and column.isdigit():
            return frame.columns[int(column)]
        if column not in set(frame.columns):
            raise ValueError(f"configured column {column!r} not found in {list(frame.columns)}")
        return column

    def _infer_mat_signal(self, mat: Dict[str, object], channel: str) -> Tuple[str, np.ndarray]:
        candidates: List[Tuple[str, np.ndarray]] = []
        for name, value in _iter_numeric_mat_arrays(mat):
            if name.startswith("__"):
                continue
            array = np.asarray(value).squeeze()
            if array.size > 1 and array.ndim <= 2:
                candidates.append((name, array))
        if not candidates:
            raise ValueError("could not infer a numeric MATLAB signal")
        channel_lower = channel.lower()
        for name, array in candidates:
            if channel_lower in name.lower():
                return name, array
        return sorted(candidates, key=lambda item: item[0])[0]

    def _resolve_mat_path(self, mat: Dict[str, object], path: str) -> object:
        value: object = mat
        for token in _mat_path_tokens(path):
            if isinstance(token, str):
                if isinstance(value, dict):
                    value = value[token]
                elif isinstance(value, np.ndarray) and value.dtype.names and token in value.dtype.names:
                    value = value[token]
                elif hasattr(value, token):
                    value = getattr(value, token)
                elif isinstance(value, np.void) and value.dtype.names and token in value.dtype.names:
                    value = value[token]
                else:
                    raise KeyError(f"could not resolve MATLAB path token {token!r} in {path!r}")
            else:
                value = np.asarray(value)[token]
        return value

    def _ensure_source_file(self, spec: Dict[str, object]) -> Path:
        configured_path = spec.get("path")
        if configured_path:
            path = self._resolve_path(Path(str(configured_path)))
            if not path.exists():
                raise FileNotFoundError(f"public bearing source file not found: {path}")
            return path

        file_name = str(spec.get("file_name", "")).strip()
        if not file_name:
            raise ValueError("trial or record must configure either path or file_name")
        cache_dir = self._resolve_path(Path(str(self.config.get("cache_dir", f"data/external/{self.source_key}"))))
        path = cache_dir / file_name
        if path.exists():
            self._check_md5(path, spec)
            return path

        url = str(spec.get("url") or "").strip()
        if not url:
            raise FileNotFoundError(
                f"{path} does not exist. Download/extract the public dataset or configure an explicit path."
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(path.suffix + ".download")
        urllib.request.urlretrieve(url, tmp_path)
        tmp_path.replace(path)
        self._check_md5(path, spec)
        return path

    def _manifest_for(
        self,
        trial: Dict[str, object],
        telemetry: pd.DataFrame,
        source_files: List[Path],
        source_variables: Dict[str, str],
    ) -> TrialManifest:
        profile = self._channel_profile()
        sample_rate_hz = self._sample_rate_hz(trial)
        duration_s = float(len(telemetry) / sample_rate_hz)
        fault_label = str(trial.get("fault_label", "none"))
        induced = fault_label != "none" and bool(trial.get("fault_induced", True))
        source_type = str(self.config.get("source_type", f"external_{self.source_key}"))
        dataset_name = str(self.config.get("dataset_name", self.source_key))
        return TrialManifest(
            trial_id=str(trial["trial_id"]),
            created_at=datetime.now(timezone.utc).isoformat(),
            operator=f"{source_type}_adapter",
            hardware={
                "motor": str(trial.get("motor", "public bearing test rig")),
                "bearing_dataset": dataset_name,
            },
            sensor_config={
                "sample_rate_hz": sample_rate_hz,
                "vibration_channels": list(profile.get("vibration_channels", [])),
                "current_channels": list(profile.get("current_channels", [])),
                "scalar_channels": list(profile.get("scalar_channels", [])),
                "adc_resolution_bits": trial.get("adc_resolution_bits"),
            },
            baseline={
                "id": int(trial.get("baseline_id", 0)),
                "name": str(trial.get("baseline_name", f"{self.source_key}_baseline")),
                "rpm": trial.get("rpm"),
                "load_hp": trial.get("load_hp"),
                "load_n": trial.get("load_n"),
            },
            fault={
                "induced": induced,
                "type": fault_label,
                "start_time_s": float(trial.get("fault_start_s", 0.0)) if induced else None,
                "end_time_s": float(trial.get("fault_end_s", duration_s)) if induced else None,
            },
            collection={
                "duration_s": duration_s,
                "source_type": source_type,
                "source_version": str(self.config.get("schema_version", "0.1.0")),
                "source_url": str(trial.get("url") or self.config.get("source_url", "")),
                "source_files": [path.name for path in source_files],
                "source_md5": {path.name: self._md5(path) for path in source_files if path.exists()},
                "source_dataset": dataset_name,
                "channel_profile": profile,
                "source_variables": source_variables,
                "start_sample": int(trial.get("start_sample", 0)),
                "reduced_profile_reason": str(
                    self.config.get(
                        "reduced_profile_reason",
                        "Public bearing dataset attached through a reduced external channel profile.",
                    )
                ),
            },
            notes=f"{dataset_name} public bearing trial; fault={fault_label}.",
        )

    def _events_for(self, trial: Dict[str, object], duration_s: float) -> List[EventRow]:
        fault_label = str(trial.get("fault_label", "none"))
        induced = fault_label != "none" and bool(trial.get("fault_induced", True))
        events = [EventRow(timestamp_s=0.0, event_type="trial_start", value=None, notes=None)]
        if induced:
            start_s = float(trial.get("fault_start_s", 0.0))
            end_s = float(trial.get("fault_end_s", duration_s))
            events.append(EventRow(timestamp_s=start_s, event_type="fault_start", value=fault_label, notes=None))
            events.append(EventRow(timestamp_s=end_s, event_type="fault_end", value=fault_label, notes=None))
        events.append(EventRow(timestamp_s=duration_s, event_type="trial_end", value=None, notes=None))
        return events

    def _notes_for(self, trial: Dict[str, object], source_files: List[Path]) -> str:
        dataset_name = str(self.config.get("dataset_name", self.source_key))
        lines = [
            f"# {trial['trial_id']}",
            "",
            f"{dataset_name} public bearing dataset trial converted to the canonical raw-trial folder layout.",
            "",
        ]
        for path in source_files:
            lines.append(f"- Source file: `{path}`")
        lines.extend(["", str(self.config.get("source_notes", "")).strip(), ""])
        return "\n".join(lines)

    def _signal_channels(self) -> List[str]:
        profile = self._channel_profile()
        return (
            list(profile.get("vibration_channels", []))
            + list(profile.get("current_channels", []))
            + list(profile.get("scalar_channels", []))
        )

    def _channel_profile(self) -> Dict[str, object]:
        profile = self.config.get("channel_profile", {})
        if not isinstance(profile, dict):
            raise ValueError("channel_profile must be a mapping")
        return profile

    def _sample_rate_hz(self, trial: Dict[str, object]) -> int:
        sampling = self.config.get("sampling", {})
        if not isinstance(sampling, dict):
            sampling = {}
        value = trial.get("sample_rate_hz", sampling.get("raw_sample_rate_hz", 12000))
        sample_rate_hz = int(value)
        if sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be positive")
        return sample_rate_hz

    def _resolve_path(self, path: Path) -> Path:
        return path if path.is_absolute() else self.project_root / path

    def _check_md5(self, path: Path, spec: Dict[str, object]) -> None:
        expected = str(spec.get("md5") or "").strip().lower()
        if expected and self._md5(path).lower() != expected:
            raise ValueError(f"MD5 mismatch for {path}: expected {expected}, got {self._md5(path)}")

    def _md5(self, path: Path) -> str:
        digest = hashlib.md5()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()


class IMSBearingSource(PublicBearingSource):
    @classmethod
    def from_config_path(cls, path: Path) -> "IMSBearingSource":
        config_path = path.resolve()
        project_root = config_path.parent.parent if config_path.parent.name == "configs" else config_path.parent
        return cls(read_yaml(config_path), project_root, "ims")


class FEMTOBearingSource(PublicBearingSource):
    @classmethod
    def from_config_path(cls, path: Path) -> "FEMTOBearingSource":
        config_path = path.resolve()
        project_root = config_path.parent.parent if config_path.parent.name == "configs" else config_path.parent
        return cls(read_yaml(config_path), project_root, "femto")


class HUSTBearingSource(PublicBearingSource):
    @classmethod
    def from_config_path(cls, path: Path) -> "HUSTBearingSource":
        config_path = path.resolve()
        project_root = config_path.parent.parent if config_path.parent.name == "configs" else config_path.parent
        return cls(read_yaml(config_path), project_root, "hust")


class PaderbornBearingSource(PublicBearingSource):
    @classmethod
    def from_config_path(cls, path: Path) -> "PaderbornBearingSource":
        config_path = path.resolve()
        project_root = config_path.parent.parent if config_path.parent.name == "configs" else config_path.parent
        return cls(read_yaml(config_path), project_root, "paderborn")


def _mat_path_tokens(path: str) -> Iterable[object]:
    for part in path.split("."):
        match = re.match(r"^([^\[]+)", part)
        if match:
            yield match.group(1)
        for index_text in re.findall(r"\[([0-9,\s]+)\]", part):
            indices = tuple(int(item.strip()) for item in index_text.split(",") if item.strip())
            yield indices[0] if len(indices) == 1 else indices


def _iter_numeric_mat_arrays(mat: Dict[str, object]) -> Iterable[Tuple[str, np.ndarray]]:
    for key, value in mat.items():
        if key.startswith("__"):
            continue
        yield from _iter_numeric_value(key, value)


def _iter_numeric_value(name: str, value: object) -> Iterable[Tuple[str, np.ndarray]]:
    if isinstance(value, np.ndarray):
        if value.dtype.names:
            for field in value.dtype.names:
                yield from _iter_numeric_value(f"{name}.{field}", value[field])
        elif np.issubdtype(value.dtype, np.number):
            yield name, value
        elif value.size <= 16:
            for index, item in np.ndenumerate(value):
                yield from _iter_numeric_value(f"{name}[{','.join(str(i) for i in index)}]", item)
    elif isinstance(value, np.void) and value.dtype.names:
        for field in value.dtype.names:
            yield from _iter_numeric_value(f"{name}.{field}", value[field])
    elif hasattr(value, "__dict__"):
        for field, item in vars(value).items():
            if not field.startswith("_"):
                yield from _iter_numeric_value(f"{name}.{field}", item)
