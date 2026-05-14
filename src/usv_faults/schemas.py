from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from typing import Any, Dict, List, Optional


def _drop_none(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _drop_none(item) for key, item in value.items() if item is not None}
    if isinstance(value, list):
        return [_drop_none(item) for item in value]
    return value


class SchemaMixin:
    @classmethod
    def model_validate(cls, data: Dict[str, Any]):
        return cls.from_dict(data)

    def model_dump(self, mode: str = "json", exclude_none: bool = False) -> Dict[str, Any]:
        del mode

        def convert(value: Any) -> Any:
            if is_dataclass(value):
                data = {
                    item.name: convert(getattr(value, item.name))
                    for item in fields(value)
                    if item.name != "model_extra"
                }
                extra = getattr(value, "model_extra", {}) or {}
                data.update({key: convert(item) for key, item in extra.items()})
                return data
            if isinstance(value, dict):
                return {key: convert(item) for key, item in value.items()}
            if isinstance(value, list):
                return [convert(item) for item in value]
            return value

        dumped = convert(self)
        return _drop_none(dumped) if exclude_none else dumped


def _known_field_names(cls: type) -> set:
    return {item.name for item in fields(cls)}


@dataclass
class HardwareConfig(SchemaMixin):
    pi_id: Optional[str] = None
    teensy_id: Optional[str] = None
    motor: Optional[str] = None
    esc: Optional[str] = None
    power_supply: Optional[str] = None
    model_extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "HardwareConfig":
        data = data or {}
        known = _known_field_names(cls) - {"model_extra"}
        return cls(
            **{key: data.get(key) for key in known},
            model_extra={key: value for key, value in data.items() if key not in known},
        )


@dataclass
class SensorConfig(SchemaMixin):
    sample_rate_hz: int
    vibration_channels: List[str] = field(default_factory=list)
    accelerometer_channels: List[str] = field(default_factory=list)
    current_channels: List[str] = field(default_factory=list)
    scalar_channels: List[str] = field(default_factory=list)
    adc_resolution_bits: Optional[int] = None

    @property
    def all_vibration_channels(self) -> List[str]:
        return self.vibration_channels or self.accelerometer_channels

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SensorConfig":
        return cls(
            sample_rate_hz=int(data["sample_rate_hz"]),
            vibration_channels=list(data.get("vibration_channels", [])),
            accelerometer_channels=list(data.get("accelerometer_channels", [])),
            current_channels=list(data.get("current_channels", [])),
            scalar_channels=list(data.get("scalar_channels", [])),
            adc_resolution_bits=data.get("adc_resolution_bits"),
        )


@dataclass
class BaselineConfig(SchemaMixin):
    id: int
    name: str
    voltage_v: Optional[float] = None
    water_temperature_c: Optional[float] = None
    model_extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BaselineConfig":
        known = _known_field_names(cls) - {"model_extra"}
        return cls(
            id=int(data["id"]),
            name=str(data["name"]),
            voltage_v=data.get("voltage_v"),
            water_temperature_c=data.get("water_temperature_c"),
            model_extra={key: value for key, value in data.items() if key not in known},
        )


@dataclass
class FaultConfig(SchemaMixin):
    induced: bool
    type: str = "none"
    start_time_s: Optional[float] = None
    end_time_s: Optional[float] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FaultConfig":
        return cls(
            induced=bool(data["induced"]),
            type=str(data.get("type", "none")),
            start_time_s=data.get("start_time_s"),
            end_time_s=data.get("end_time_s"),
        )


@dataclass
class CollectionConfig(SchemaMixin):
    duration_s: float
    source_type: str
    source_version: str = "0.1.0"
    random_seed: Optional[int] = None
    model_extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CollectionConfig":
        known = _known_field_names(cls) - {"model_extra"}
        return cls(
            duration_s=float(data["duration_s"]),
            source_type=str(data["source_type"]),
            source_version=str(data.get("source_version", "0.1.0")),
            random_seed=data.get("random_seed"),
            model_extra={key: value for key, value in data.items() if key not in known},
        )


@dataclass
class TrialManifest(SchemaMixin):
    trial_id: str
    created_at: str
    sensor_config: SensorConfig
    baseline: BaselineConfig
    fault: FaultConfig
    collection: CollectionConfig
    operator: str = "synthetic"
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    notes: str = ""
    model_extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrialManifest":
        known = _known_field_names(cls) - {"model_extra"}
        return cls(
            trial_id=str(data["trial_id"]),
            created_at=str(data["created_at"]),
            operator=str(data.get("operator", "synthetic")),
            hardware=HardwareConfig.from_dict(data.get("hardware", {})),
            sensor_config=SensorConfig.from_dict(data["sensor_config"]),
            baseline=BaselineConfig.from_dict(data["baseline"]),
            fault=FaultConfig.from_dict(data["fault"]),
            collection=CollectionConfig.from_dict(data["collection"]),
            notes=str(data.get("notes", "")),
            model_extra={key: value for key, value in data.items() if key not in known},
        )


@dataclass
class EventRow(SchemaMixin):
    timestamp_s: float
    event_type: str
    value: Optional[str] = None
    notes: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EventRow":
        return cls(
            timestamp_s=float(data["timestamp_s"]),
            event_type=str(data["event_type"]),
            value=data.get("value"),
            notes=data.get("notes"),
        )


@dataclass
class QualityReport(SchemaMixin):
    trial_id: str
    status: str
    sample_count: int
    duration_s: float
    sample_rate_hz_estimate: float
    channel_checks: Dict[str, Any]
    event_checks: Dict[str, Any]
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QualityReport":
        return cls(**data)


@dataclass
class DatasetManifest(SchemaMixin):
    dataset_id: str
    source_trials: List[str]
    status: str = "active"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DatasetManifest":
        return cls(
            dataset_id=str(data["dataset_id"]),
            source_trials=list(data["source_trials"]),
            status=str(data.get("status", "active")),
        )


@dataclass
class TrainingRunManifest(SchemaMixin):
    run_id: str
    run_type: str
    dataset_id: str

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainingRunManifest":
        return cls(**data)


@dataclass
class DictionaryManifest(SchemaMixin):
    dictionary_id: str
    source_model: str
    source_dataset: str

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DictionaryManifest":
        return cls(**data)


@dataclass
class ChannelProfile(SchemaMixin):
    name: str
    expected_input_dim: int
    vibration_channels: List[str]
    current_channels: List[str]
    scalar_channels: List[str]
    scalar_features: List[str]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChannelProfile":
        return cls(
            name=str(data["name"]),
            expected_input_dim=int(data["expected_input_dim"]),
            vibration_channels=list(data["vibration_channels"]),
            current_channels=list(data["current_channels"]),
            scalar_channels=list(data["scalar_channels"]),
            scalar_features=list(data["scalar_features"]),
        )


@dataclass
class SyntheticBaseline(SchemaMixin):
    description: str
    voltage_v: float
    water_temperature_c: float
    drag_multiplier: float = 1.0
    vibration_noise_multiplier: float = 1.0
    pwm_profile: str = "steady"
    model_extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SyntheticBaseline":
        known = _known_field_names(cls) - {"model_extra"}
        return cls(
            description=str(data["description"]),
            voltage_v=float(data["voltage_v"]),
            water_temperature_c=float(data["water_temperature_c"]),
            drag_multiplier=float(data.get("drag_multiplier", 1.0)),
            vibration_noise_multiplier=float(data.get("vibration_noise_multiplier", 1.0)),
            pwm_profile=str(data.get("pwm_profile", "steady")),
            model_extra={key: value for key, value in data.items() if key not in known},
        )


@dataclass
class SyntheticTrial(SchemaMixin):
    trial_id: str
    baseline: str
    fault: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SyntheticTrial":
        return cls(
            trial_id=str(data["trial_id"]),
            baseline=str(data["baseline"]),
            fault=data.get("fault"),
        )


@dataclass
class SyntheticTrialSet(SchemaMixin):
    duration_s: float
    fault_start_s: Optional[float] = None
    fault_end_s: Optional[float] = None
    trials: List[SyntheticTrial] = field(default_factory=list)
    baselines: List[str] = field(default_factory=list)
    faults: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SyntheticTrialSet":
        return cls(
            duration_s=float(data["duration_s"]),
            fault_start_s=data.get("fault_start_s"),
            fault_end_s=data.get("fault_end_s"),
            trials=[SyntheticTrial.from_dict(item) for item in data.get("trials", [])],
            baselines=list(data.get("baselines", [])),
            faults=list(data.get("faults", [])),
        )


@dataclass
class SyntheticConfig(SchemaMixin):
    attachment_id: str
    source_type: str
    schema_version: Any
    output: Dict[str, Any]
    randomness: Dict[str, Any]
    sampling: Dict[str, Any]
    channel_profile: ChannelProfile
    nominal_motor: Dict[str, Any]
    baselines: Dict[str, SyntheticBaseline]
    fault_profiles: Dict[str, Dict[str, Any]]
    trial_sets: Dict[str, SyntheticTrialSet]
    splits: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SyntheticConfig":
        return cls(
            attachment_id=str(data["attachment_id"]),
            source_type=str(data["source_type"]),
            schema_version=data["schema_version"],
            output=dict(data["output"]),
            randomness=dict(data["randomness"]),
            sampling=dict(data["sampling"]),
            channel_profile=ChannelProfile.from_dict(data["channel_profile"]),
            nominal_motor=dict(data["nominal_motor"]),
            baselines={
                key: SyntheticBaseline.from_dict(value) for key, value in data["baselines"].items()
            },
            fault_profiles={key: dict(value) for key, value in data["fault_profiles"].items()},
            trial_sets={
                key: SyntheticTrialSet.from_dict(value) for key, value in data["trial_sets"].items()
            },
            splits=dict(data.get("splits", {})),
        )

    def model_dump(self, mode: str = "json", exclude_none: bool = False) -> Dict[str, Any]:
        del mode

        def convert(value: Any) -> Any:
            if is_dataclass(value):
                return value.model_dump(exclude_none=exclude_none)
            if isinstance(value, dict):
                return {key: convert(item) for key, item in value.items()}
            if isinstance(value, list):
                return [convert(item) for item in value]
            return value

        dumped = {field_name.name: convert(getattr(self, field_name.name)) for field_name in fields(self)}
        return _drop_none(dumped) if exclude_none else dumped
