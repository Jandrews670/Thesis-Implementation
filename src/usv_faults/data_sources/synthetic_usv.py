from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from usv_faults.config import load_config
from usv_faults.schemas import EventRow, SyntheticConfig, SyntheticTrial, TrialManifest
from usv_faults.storage.trials import REQUIRED_TRIAL_FILES, quality_check_trial, write_events, write_manifest


def _stable_trial_seed(master_seed: int, trial_id: str) -> int:
    digest = hashlib.sha256(f"{master_seed}:{trial_id}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


class SyntheticUSVSource:
    def __init__(self, config: SyntheticConfig) -> None:
        self.config = config

    @classmethod
    def from_config_path(cls, path: Path) -> "SyntheticUSVSource":
        return cls(load_config(path, SyntheticConfig))

    def attach(self, out_dir: Path) -> List[Path]:
        out_dir.mkdir(parents=True, exist_ok=True)
        trial_specs = self._expand_trial_specs()
        created: List[Path] = []
        for trial, duration_s, fault_start_s, fault_end_s in trial_specs:
            trial_dir = out_dir / trial.trial_id
            trial_dir.mkdir(parents=True, exist_ok=True)
            if all((trial_dir / filename).exists() for filename in REQUIRED_TRIAL_FILES):
                quality_check_trial(trial_dir)
                created.append(trial_dir)
                continue
            seed = _stable_trial_seed(int(self.config.randomness["master_seed"]), trial.trial_id)
            telemetry = self._generate_trial(trial, duration_s, fault_start_s, fault_end_s, seed)
            telemetry.to_parquet(trial_dir / "telemetry.parquet", index=False)
            manifest = self._manifest_for(trial, duration_s, fault_start_s, fault_end_s, seed)
            write_manifest(trial_dir / "manifest.yaml", manifest)
            write_events(trial_dir / "events.csv", self._events_for(trial, duration_s, fault_start_s, fault_end_s))
            (trial_dir / "notes.md").write_text(
                f"# {trial.trial_id}\n\nSynthetic proof-of-concept trial.\n",
                encoding="utf-8",
            )
            quality_check_trial(trial_dir)
            created.append(trial_dir)
        return created

    def _expand_trial_specs(self) -> List[Tuple[SyntheticTrial, float, Optional[float], Optional[float]]]:
        specs: List[Tuple[SyntheticTrial, float, Optional[float], Optional[float]]] = []
        for trial_set in self.config.trial_sets.values():
            for trial in trial_set.trials:
                specs.append((trial, trial_set.duration_s, trial_set.fault_start_s, trial_set.fault_end_s))
            for baseline_name in trial_set.baselines:
                for fault_name in trial_set.faults:
                    fault_slug = fault_name.replace("propeller_", "").replace("bearing_", "bearing_")
                    trial_id = f"2026-05-14_POC_{baseline_name}_fault_{fault_slug}_T001"
                    specs.append(
                        (
                            SyntheticTrial(trial_id=trial_id, baseline=baseline_name, fault=fault_name),
                            trial_set.duration_s,
                            trial_set.fault_start_s,
                            trial_set.fault_end_s,
                        )
                    )
        return specs

    def _generate_trial(
        self,
        trial: SyntheticTrial,
        duration_s: float,
        fault_start_s: Optional[float],
        fault_end_s: Optional[float],
        seed: int,
    ) -> pd.DataFrame:
        sampling = self.config.sampling
        sample_rate_hz = int(sampling["raw_sample_rate_hz"])
        sample_count = int(round(duration_s * sample_rate_hz))
        timestamp_s = np.arange(sample_count, dtype=np.float64) / sample_rate_hz
        rng = np.random.default_rng(seed)

        baseline = self.config.baselines[trial.baseline]
        nominal = self.config.nominal_motor
        shaft_hz = float(nominal["shaft_frequency_hz"])
        base_pwm = float(nominal["pwm_command"])

        pwm = np.full(sample_count, base_pwm, dtype=np.float64)
        if baseline.pwm_profile == "ventilation":
            ventilation = baseline.model_extra.get("ventilation", {})
            interval_s = float(ventilation.get("interval_s", 5.0))
            drop_pwm = float(ventilation.get("drop_pwm", 0.10))
            surge_pwm = float(ventilation.get("surge_pwm", 0.90))
            drop_range = ventilation.get("drop_duration_s_range", [0.5, 1.5])
            surge_duration_s = float(ventilation.get("surge_duration_s", 0.5))
            start_s = interval_s
            while start_s < duration_s:
                drop_duration_s = float(rng.uniform(float(drop_range[0]), float(drop_range[1])))
                drop_mask = (timestamp_s >= start_s) & (timestamp_s < start_s + drop_duration_s)
                surge_mask = (timestamp_s >= start_s + drop_duration_s) & (
                    timestamp_s < start_s + drop_duration_s + surge_duration_s
                )
                pwm[drop_mask] = drop_pwm
                pwm[surge_mask] = surge_pwm
                start_s += interval_s

        voltage = np.full(sample_count, baseline.voltage_v, dtype=np.float64)
        temperature = np.full(sample_count, baseline.water_temperature_c, dtype=np.float64)
        voltage += 0.02 * rng.standard_normal(sample_count)
        temperature += 0.01 * rng.standard_normal(sample_count)

        effective_shaft_hz = shaft_hz * (0.85 + 0.3 * pwm) * (baseline.voltage_v / 16.0)
        phase = 2 * np.pi * np.cumsum(effective_shaft_hz) / sample_rate_hz
        drag = baseline.drag_multiplier
        noise = baseline.vibration_noise_multiplier

        motor_vibration = (
            0.8 * drag * np.sin(phase)
            + 0.28 * np.sin(2 * phase + 0.5)
            + 0.12 * np.sin(3 * phase + 1.2)
            + 0.05 * noise * rng.standard_normal(sample_count)
        )
        rig_vibration = (
            0.22 * np.sin(phase + 0.2)
            + 0.12 * np.sin(2 * np.pi * 8.0 * timestamp_s)
            + 0.04 * noise * rng.standard_normal(sample_count)
        )
        motor_current = (
            2.0
            + 1.5 * pwm
            + 0.45 * (drag - 1.0)
            + 0.08 * np.sin(phase + 0.8)
            + 0.03 * rng.standard_normal(sample_count)
        )

        if baseline.pwm_profile == "ventilation":
            disturbance = 0.35 * np.sin(2 * np.pi * 5.0 * timestamp_s) * (pwm != base_pwm)
            motor_vibration += disturbance
            rig_vibration += 0.5 * disturbance
            motor_current += 0.4 * np.abs(pwm - base_pwm)

        fault_mask = np.zeros(sample_count, dtype=bool)
        if trial.fault and fault_start_s is not None and fault_end_s is not None:
            fault_mask = (timestamp_s >= fault_start_s) & (timestamp_s <= fault_end_s)
            self._apply_fault(
                trial.fault,
                timestamp_s,
                phase,
                fault_mask,
                rng,
                motor_vibration,
                rig_vibration,
                motor_current,
            )

        return pd.DataFrame(
            {
                "timestamp_s": timestamp_s,
                "motor_vibration": motor_vibration,
                "rig_vibration": rig_vibration,
                "motor_current": motor_current,
                "voltage": voltage,
                "water_temperature": temperature,
                "pwm_command": pwm,
            }
        )

    def _apply_fault(
        self,
        fault: str,
        timestamp_s: np.ndarray,
        phase: np.ndarray,
        fault_mask: np.ndarray,
        rng: np.random.Generator,
        motor_vibration: np.ndarray,
        rig_vibration: np.ndarray,
        motor_current: np.ndarray,
    ) -> None:
        if fault == "bearing_impulse":
            impulse_train = ((timestamp_s * 95.0) % 1.0) < 0.025
            impulses = fault_mask & impulse_train
            motor_vibration[impulses] += 2.0 + 0.4 * rng.standard_normal(np.count_nonzero(impulses))
            motor_vibration[fault_mask] += 0.35 * np.sin(6.5 * phase[fault_mask])
            rig_vibration[fault_mask] += 0.12 * np.sin(6.5 * phase[fault_mask])
            motor_current[fault_mask] += 0.04 * np.sin(2.5 * phase[fault_mask])
        elif fault == "propeller_imbalance":
            modulation = 1.0 + 0.45 * np.sin(2 * np.pi * 1.2 * timestamp_s[fault_mask])
            motor_vibration[fault_mask] += modulation * (0.75 * np.sin(phase[fault_mask]))
            rig_vibration[fault_mask] += modulation * (0.28 * np.sin(phase[fault_mask] + 0.4))
            motor_current[fault_mask] += 0.18 * modulation
        elif fault == "shaft_rub":
            bursts = fault_mask & (((timestamp_s * 7.0) % 1.0) < 0.2)
            motor_vibration[bursts] += 0.8 * rng.standard_normal(np.count_nonzero(bursts))
            rig_vibration[bursts] += 0.35 * rng.standard_normal(np.count_nonzero(bursts))
            motor_current[fault_mask] += 0.5 + 0.12 * np.sin(4 * phase[fault_mask])
        elif fault == "electrical_phase_noise":
            motor_current[fault_mask] += 0.55 * np.sin(5.0 * phase[fault_mask])
            motor_current[fault_mask] += 0.12 * rng.standard_normal(np.count_nonzero(fault_mask))
            motor_vibration[fault_mask] += 0.08 * np.sin(5.0 * phase[fault_mask])

    def _manifest_for(
        self,
        trial: SyntheticTrial,
        duration_s: float,
        fault_start_s: Optional[float],
        fault_end_s: Optional[float],
        seed: int,
    ) -> TrialManifest:
        baseline = self.config.baselines[trial.baseline]
        baseline_id = int(trial.baseline[1]) if trial.baseline.startswith("B") else -1
        profile = self.config.channel_profile
        return TrialManifest(
            trial_id=trial.trial_id,
            created_at=datetime.now(timezone.utc).isoformat(),
            operator="synthetic",
            hardware={
                "pi_id": "synthetic",
                "teensy_id": "synthetic",
                "motor": "synthetic_aspiqueen_u01",
                "esc": "synthetic",
                "power_supply": "synthetic",
            },
            sensor_config={
                "sample_rate_hz": int(self.config.sampling["raw_sample_rate_hz"]),
                "vibration_channels": profile.vibration_channels,
                "current_channels": profile.current_channels,
                "scalar_channels": profile.scalar_channels,
                "adc_resolution_bits": None,
            },
            baseline={
                "id": baseline_id,
                "name": trial.baseline,
                "voltage_v": baseline.voltage_v,
                "water_temperature_c": baseline.water_temperature_c,
            },
            fault={
                "induced": bool(trial.fault),
                "type": trial.fault or "none",
                "start_time_s": fault_start_s if trial.fault else None,
                "end_time_s": fault_end_s if trial.fault else None,
            },
            collection={
                "duration_s": duration_s,
                "source_type": self.config.source_type,
                "source_version": str(self.config.schema_version),
                "random_seed": seed,
            },
            notes=f"Synthetic baseline {trial.baseline}; fault={trial.fault or 'none'}.",
        )

    def _events_for(
        self,
        trial: SyntheticTrial,
        duration_s: float,
        fault_start_s: Optional[float],
        fault_end_s: Optional[float],
    ) -> List[EventRow]:
        events = [
            EventRow(timestamp_s=0.0, event_type="trial_start", value=None, notes=None),
        ]
        if trial.fault and fault_start_s is not None and fault_end_s is not None:
            events.extend(
                [
                    EventRow(
                        timestamp_s=fault_start_s,
                        event_type="fault_start",
                        value=trial.fault,
                        notes=None,
                    ),
                    EventRow(
                        timestamp_s=fault_end_s,
                        event_type="fault_end",
                        value=trial.fault,
                        notes=None,
                    ),
                ]
            )
        events.append(EventRow(timestamp_s=duration_s, event_type="trial_end", value=None, notes=None))
        return events
