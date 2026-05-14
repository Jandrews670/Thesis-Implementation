from __future__ import annotations

import copy
import unittest
from pathlib import Path

import pandas as pd

from usv_faults.config import load_config
from usv_faults.data_sources.synthetic_usv import SyntheticUSVSource
from usv_faults.schemas import SyntheticConfig
from usv_faults.storage.trials import quality_check_trial, read_manifest


class ObjectiveOneTests(unittest.TestCase):
    def test_synthetic_config_loads(self) -> None:
        config = load_config(Path("configs/poc_synthetic.yaml"), SyntheticConfig)

        self.assertEqual(config.channel_profile.expected_input_dim, 2109)
        self.assertEqual(config.channel_profile.vibration_channels, ["motor_vibration", "rig_vibration"])

    def test_synthetic_source_writes_trial_and_qc(self) -> None:
        import tempfile

        config = load_config(Path("configs/poc_synthetic.yaml"), SyntheticConfig)
        reduced = copy.deepcopy(config)
        reduced.trial_sets = {
            "healthy_training": copy.deepcopy(reduced.trial_sets["healthy_training"])
        }
        reduced.trial_sets["healthy_training"].duration_s = 1.0
        reduced.trial_sets["healthy_training"].trials = [
            reduced.trial_sets["healthy_training"].trials[0]
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            created = SyntheticUSVSource(reduced).attach(Path(temp_dir))

            self.assertEqual(len(created), 1)
            trial_dir = created[0]
            self.assertTrue((trial_dir / "manifest.yaml").exists())
            self.assertTrue((trial_dir / "telemetry.parquet").exists())
            self.assertTrue((trial_dir / "events.csv").exists())
            self.assertTrue((trial_dir / "quality_report.json").exists())

            manifest = read_manifest(trial_dir)
            telemetry = pd.read_parquet(trial_dir / "telemetry.parquet")
            report = quality_check_trial(trial_dir)

            self.assertEqual(manifest.trial_id, "2026-05-14_POC_B0_nominal_T001")
            self.assertEqual(len(telemetry), 10000)
            self.assertIn(report.status, {"accepted", "accepted_with_notes"})


if __name__ == "__main__":
    unittest.main()
