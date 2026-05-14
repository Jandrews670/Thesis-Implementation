from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from usv_faults.config import load_config, read_yaml, write_yaml
from usv_faults.data_sources.synthetic_usv import SyntheticUSVSource
from usv_faults.preprocessing.datasets import make_dataset
from usv_faults.schemas import SyntheticConfig
from usv_faults.training.train_sdae import train_sdae


class ObjectiveThreeTests(unittest.TestCase):
    def test_train_sdae_writes_required_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            raw_root = root / "raw"
            dataset_dir = root / "dataset"
            model_dir = root / "model"

            source_config = load_config(Path("configs/poc_synthetic_training_smoke.yaml"), SyntheticConfig)
            reduced_source = copy.deepcopy(source_config)
            for trial_set in reduced_source.trial_sets.values():
                trial_set.duration_s = 1.0
                if trial_set.fault_start_s is not None:
                    trial_set.fault_start_s = 0.2
                    trial_set.fault_end_s = 0.8
            SyntheticUSVSource(reduced_source).attach(raw_root)

            dataset_config = copy.deepcopy(read_yaml(Path("configs/dataset_poc_synthetic_training_smoke.yaml")))
            dataset_config["raw_trial_root"] = str(raw_root)
            dataset_config_path = root / "dataset.yaml"
            write_yaml(dataset_config_path, dataset_config)
            make_dataset(dataset_config_path, dataset_dir)

            train_config = copy.deepcopy(read_yaml(Path("configs/baseline_sdae_smoke.yaml")))
            train_config["model"]["hidden_dims"] = [32]
            train_config["model"]["latent_dim"] = 8
            train_config["model"]["hidden_activation"] = "relu"
            train_config["model"]["output_activation"] = "sigmoid"
            train_config["training"]["epochs"] = 2
            train_config["training"]["batch_size"] = 4
            train_config_path = root / "train.yaml"
            write_yaml(train_config_path, train_config)

            result = train_sdae(dataset_dir, train_config_path, model_dir)

            self.assertEqual(result["train_windows"], 10)
            self.assertTrue((model_dir / "run_manifest.yaml").exists())
            self.assertTrue((model_dir / "model.pt").exists())
            self.assertTrue((model_dir / "scaler.joblib").exists())
            self.assertTrue((model_dir / "threshold.json").exists())
            self.assertTrue((model_dir / "training_history.csv").exists())
            self.assertTrue((model_dir / "metrics.json").exists())
            self.assertTrue((model_dir / "plots" / "loss_curve.png").exists())
            self.assertTrue((model_dir / "plots" / "reconstruction_error_hist.png").exists())

            run_manifest = read_yaml(model_dir / "run_manifest.yaml")
            self.assertEqual(run_manifest["model"]["hidden_activation"], "relu")
            self.assertEqual(run_manifest["model"]["output_activation"], "sigmoid")

            with (model_dir / "threshold.json").open("r", encoding="utf-8") as handle:
                threshold = json.load(handle)
            self.assertGreater(threshold["threshold"], 0.0)


if __name__ == "__main__":
    unittest.main()
