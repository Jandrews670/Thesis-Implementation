from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from usv_faults.config import load_config, read_yaml, write_yaml
from usv_faults.data_sources.synthetic_usv import SyntheticUSVSource
from usv_faults.preprocessing.datasets import make_dataset
from usv_faults.preprocessing.windowing import WindowingConfig, build_windows_for_trial
from usv_faults.schemas import SyntheticConfig
from usv_faults.storage.preview import write_preview_csv
from usv_faults.storage.trials import read_events, read_manifest


class ObjectiveTwoTests(unittest.TestCase):
    def _write_smoke_trials(self, root: Path) -> None:
        config = load_config(Path("configs/poc_synthetic_smoke.yaml"), SyntheticConfig)
        SyntheticUSVSource(config).attach(root)

    def test_windowing_builds_2109_dimensional_rows_and_fault_labels(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            raw_root = Path(temp_dir)
            self._write_smoke_trials(raw_root)
            fault_trial = raw_root / "2026-05-14_POC_B0_fault_bearing_T001"
            manifest = read_manifest(fault_trial)
            events = read_events(fault_trial)
            telemetry = pd.read_parquet(fault_trial / "telemetry.parquet")

            windows, labels = build_windows_for_trial(
                telemetry,
                manifest,
                events,
                WindowingConfig(
                    window_ms=100,
                    stride_ms=100,
                    current_sample_rate_hz=1000,
                    scalar_features=["mean", "variance", "peak_to_peak"],
                    expected_input_dim=2109,
                ),
            )

            self.assertEqual(windows.shape[1], 2109)
            self.assertEqual(len(windows), len(labels))
            self.assertTrue(labels["is_fault"].any())
            self.assertIn("bearing_impulse", set(labels["fault_label"]))

    def test_preview_and_make_dataset_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            raw_root = root / "raw"
            out_dir = root / "dataset"
            self._write_smoke_trials(raw_root)

            preview_path = write_preview_csv(raw_root / "2026-05-14_POC_B0_nominal_T001")
            self.assertTrue(preview_path.exists())
            self.assertTrue(preview_path.with_name(preview_path.stem + "_summary.csv").exists())

            config = copy.deepcopy(read_yaml(Path("configs/dataset_poc_synthetic_smoke.yaml")))
            config["raw_trial_root"] = str(raw_root)
            config_path = root / "dataset_config.yaml"
            write_yaml(config_path, config)
            result = make_dataset(config_path, out_dir)

            self.assertEqual(result["input_dim"], 2109)
            self.assertTrue((out_dir / "dataset_manifest.yaml").exists())
            self.assertTrue((out_dir / "windows.parquet").exists())
            self.assertTrue((out_dir / "labels.parquet").exists())
            labels = pd.read_parquet(out_dir / "labels.parquet")
            windows = pd.read_parquet(out_dir / "windows.parquet")
            self.assertEqual(len(labels), len(windows))
            self.assertIn("train", set(labels["split"]))
            self.assertIn("test", set(labels["split"]))


if __name__ == "__main__":
    unittest.main()
