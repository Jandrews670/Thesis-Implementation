from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from scipy.io import savemat

from usv_faults.config import write_yaml
from usv_faults.data_sources.public_bearing import (
    FEMTOBearingSource,
    HUSTBearingSource,
    IMSBearingSource,
    PaderbornBearingSource,
    PublicBearingSource,
)
from usv_faults.storage.trials import quality_check_trial, read_events, read_manifest


class PublicBearingSourceTests(unittest.TestCase):
    def test_public_bearing_sources_write_canonical_raw_trials(self) -> None:
        cases = [
            ("ims", IMSBearingSource, _ims_fixture),
            ("femto", FEMTOBearingSource, _femto_fixture),
            ("hust", HUSTBearingSource, _hust_fixture),
            ("paderborn", PaderbornBearingSource, _paderborn_fixture),
        ]
        for key, source_class, builder in cases:
            with self.subTest(source=key):
                with tempfile.TemporaryDirectory() as temp_dir:
                    root = Path(temp_dir)
                    source_dir = root / "external" / key
                    raw_root = root / "raw"
                    source_dir.mkdir(parents=True)

                    config = builder(source_dir)
                    config_path = root / f"public_{key}.yaml"
                    write_yaml(config_path, config)

                    created = source_class.from_config_path(config_path).attach(raw_root)

                    self.assertEqual(len(created), 2)
                    healthy_dir = raw_root / f"{key}_healthy"
                    fault_dir = raw_root / f"{key}_fault"
                    for trial_dir in (healthy_dir, fault_dir):
                        report = quality_check_trial(trial_dir)
                        self.assertIn(report.status, {"accepted", "accepted_with_notes"})
                        manifest = read_manifest(trial_dir)
                        telemetry = pd.read_parquet(trial_dir / "telemetry.parquet")
                        self.assertEqual(manifest.collection.source_type, f"external_{key}")
                        self.assertEqual(manifest.collection.model_extra["source_dataset"], f"fixture_{key}")
                        self.assertEqual(len(telemetry), 1000)
                        self.assertIn("timestamp_s", telemetry.columns)
                        for channel in config["channel_profile"]["vibration_channels"]:
                            self.assertIn(channel, telemetry.columns)
                        for channel in config["channel_profile"]["current_channels"]:
                            self.assertIn(channel, telemetry.columns)

                    fault_manifest = read_manifest(fault_dir)
                    fault_events = read_events(fault_dir)
                    self.assertTrue(fault_manifest.fault.induced)
                    self.assertEqual(fault_manifest.fault.type, f"{key}_fault_label")
                    self.assertIn("fault_start", {event.event_type for event in fault_events})
                    self.assertIn("fault_end", {event.event_type for event in fault_events})

    def test_public_bearing_cli_source_names_are_registered(self) -> None:
        from usv_faults.cli import build_parser

        parser = build_parser()
        for source in ("ims", "femto", "hust", "paderborn"):
            args = parser.parse_args(
                [
                    "attach-data",
                    "--source",
                    source,
                    "--config",
                    "configs/example.yaml",
                    "--out",
                    "data/raw/example",
                ]
            )
            self.assertEqual(args.source, source)

    def test_public_bearing_record_ranges_expand_to_record_paths(self) -> None:
        source = PublicBearingSource(
            {
                "channel_profile": {
                    "vibration_channels": ["vibration"],
                    "current_channels": [],
                    "scalar_channels": [],
                }
            },
            Path("."),
            "fixture",
        )

        records = source._expanded_records(
            {
                "trial_id": "range_trial",
                "record_ranges": [
                    {
                        "path_template": "records/acc_{index:05d}.csv",
                        "start": 8,
                        "end": 10,
                        "columns": {"vibration": 4},
                    }
                ],
            }
        )

        self.assertEqual(
            [record["path"] for record in records],
            ["records/acc_00008.csv", "records/acc_00009.csv", "records/acc_00010.csv"],
        )
        self.assertEqual(records[0]["columns"], {"vibration": 4})


def _base_config(key: str, source_dir: Path, channels: Dict[str, object]) -> Dict[str, object]:
    signal_channels = channels["vibration_channels"] + channels["current_channels"]
    return {
        "attachment_id": f"fixture_{key}",
        "source_type": f"external_{key}",
        "dataset_name": f"fixture_{key}",
        "schema_version": "0.1.0",
        "source_url": "https://example.invalid/public-bearing-fixture",
        "cache_dir": str(source_dir),
        "sampling": {"raw_sample_rate_hz": 1000},
        "channel_profile": {
            "name": f"{key}_fixture_profile",
            "expected_input_dim": 100 * len(signal_channels),
            "vibration_channels": channels["vibration_channels"],
            "current_channels": channels["current_channels"],
            "scalar_channels": [],
            "scalar_features": [],
        },
    }


def _healthy_signal() -> np.ndarray:
    sample_rate_hz = 1000
    t = np.arange(sample_rate_hz, dtype=np.float64) / sample_rate_hz
    return 0.25 * np.sin(2.0 * np.pi * 35.0 * t)


def _fault_signal() -> np.ndarray:
    sample_rate_hz = 1000
    t = np.arange(sample_rate_hz, dtype=np.float64) / sample_rate_hz
    impulses = ((t * 80.0) % 1.0) < 0.05
    return _healthy_signal() + 0.8 * impulses.astype(np.float64)


def _write_trials(config: Dict[str, object], key: str, paths: Dict[str, Path]) -> Dict[str, object]:
    config["trials"] = [
        {
            "trial_id": f"{key}_healthy",
            "path": str(paths["healthy"]),
            "baseline_id": 0,
            "baseline_name": f"{key}_nominal",
            "fault_label": "none",
            "fault_induced": False,
        },
        {
            "trial_id": f"{key}_fault",
            "path": str(paths["fault"]),
            "baseline_id": 0,
            "baseline_name": f"{key}_faulted",
            "fault_label": f"{key}_fault_label",
            "fault_induced": True,
            "fault_start_s": 0.2,
            "fault_end_s": 0.9,
        },
    ]
    return config


def _ims_fixture(source_dir: Path) -> Dict[str, object]:
    channels = {"vibration_channels": ["bearing_vibration"], "current_channels": []}
    config = _base_config("ims", source_dir, channels)
    config.update({"format": "delimited", "delimiter": "auto", "has_header": False, "columns": {"bearing_vibration": 0}})
    paths = {"healthy": source_dir / "healthy.txt", "fault": source_dir / "fault.txt"}
    np.savetxt(paths["healthy"], _healthy_signal().reshape(-1, 1))
    np.savetxt(paths["fault"], _fault_signal().reshape(-1, 1))
    return _write_trials(config, "ims", paths)


def _femto_fixture(source_dir: Path) -> Dict[str, object]:
    channels = {
        "vibration_channels": ["horizontal_acceleration", "vertical_acceleration"],
        "current_channels": [],
    }
    config = _base_config("femto", source_dir, channels)
    config.update(
        {
            "format": "csv",
            "delimiter": ";",
            "has_header": False,
            "columns": {"horizontal_acceleration": 4, "vertical_acceleration": 5},
        }
    )
    paths = {"healthy": source_dir / "healthy.csv", "fault": source_dir / "fault.csv"}
    for path, signal in ((paths["healthy"], _healthy_signal()), (paths["fault"], _fault_signal())):
        frame = np.column_stack([np.zeros((1000, 4)), signal, 0.5 * signal])
        np.savetxt(path, frame, delimiter=";")
    return _write_trials(config, "femto", paths)


def _hust_fixture(source_dir: Path) -> Dict[str, object]:
    channels = {"vibration_channels": ["vibration"], "current_channels": []}
    config = _base_config("hust", source_dir, channels)
    config.update({"format": "csv", "delimiter": "auto", "has_header": True, "columns": {"vibration": "vibration"}})
    paths = {"healthy": source_dir / "healthy.csv", "fault": source_dir / "fault.csv"}
    pd.DataFrame({"vibration": _healthy_signal()}).to_csv(paths["healthy"], index=False)
    pd.DataFrame({"vibration": _fault_signal()}).to_csv(paths["fault"], index=False)
    return _write_trials(config, "hust", paths)


def _paderborn_fixture(source_dir: Path) -> Dict[str, object]:
    channels = {"vibration_channels": ["vibration"], "current_channels": ["motor_current"]}
    config = _base_config("paderborn", source_dir, channels)
    config.update(
        {
            "format": "mat",
            "mat_variables": {"vibration": "vibration", "motor_current": "motor_current"},
        }
    )
    paths = {"healthy": source_dir / "healthy.mat", "fault": source_dir / "fault.mat"}
    for path, signal in ((paths["healthy"], _healthy_signal()), (paths["fault"], _fault_signal())):
        savemat(path, {"vibration": signal.reshape(-1, 1), "motor_current": (0.2 * signal).reshape(-1, 1)})
    return _write_trials(config, "paderborn", paths)


if __name__ == "__main__":
    unittest.main()
