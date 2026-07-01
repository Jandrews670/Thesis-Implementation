from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from usv_faults.clustering.fault_dictionary import build_fault_dictionary, decide_latent_cluster
from usv_faults.clustering.mahalanobis import chi_square_threshold, covariance_with_ledoit_wolf
from usv_faults.config import load_config, read_yaml, write_yaml
from usv_faults.data_sources.synthetic_usv import SyntheticUSVSource
from usv_faults.preprocessing.datasets import make_dataset
from usv_faults.schemas import SyntheticConfig
from usv_faults.training.train_sdae import train_sdae


class ObjectiveFourTests(unittest.TestCase):
    def test_ledoit_wolf_and_chi_square_outputs(self) -> None:
        import numpy as np

        rng = np.random.default_rng(20260514)
        samples = rng.normal(size=(20, 8))
        estimate = covariance_with_ledoit_wolf(samples)
        threshold = chi_square_threshold(420, 0.99)

        self.assertEqual(estimate.estimator, "sklearn.covariance.LedoitWolf")
        self.assertEqual(estimate.covariance.shape, (8, 8))
        self.assertEqual(estimate.precision.shape, (8, 8))
        self.assertGreaterEqual(estimate.shrinkage, 0.0)
        self.assertLessEqual(estimate.shrinkage, 1.0)
        self.assertEqual(threshold["method"], "scipy.stats.chi2.ppf")
        self.assertGreater(threshold["threshold"], 480.0)

    def test_build_dictionary_writes_hdbscan_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            raw_root = root / "raw"
            dataset_dir = root / "dataset"
            model_dir = root / "model"
            dictionary_dir = root / "dictionary"

            source_config = load_config(Path("configs/poc_synthetic_training_smoke.yaml"), SyntheticConfig)
            reduced_source = copy.deepcopy(source_config)
            for trial_set in reduced_source.trial_sets.values():
                trial_set.duration_s = 2.0
                if trial_set.fault_start_s is not None:
                    trial_set.fault_start_s = 0.4
                    trial_set.fault_end_s = 1.8
            SyntheticUSVSource(reduced_source).attach(raw_root)

            dataset_config = copy.deepcopy(read_yaml(Path("configs/dataset_poc_synthetic_training_smoke.yaml")))
            dataset_config["raw_trial_root"] = str(raw_root)
            dataset_config_path = root / "dataset.yaml"
            write_yaml(dataset_config_path, dataset_config)
            make_dataset(dataset_config_path, dataset_dir)

            train_config = copy.deepcopy(read_yaml(Path("configs/baseline_sdae_smoke.yaml")))
            train_config["model"]["hidden_dims"] = [32]
            train_config["model"]["latent_dim"] = 8
            train_config["training"]["epochs"] = 2
            train_config["training"]["batch_size"] = 4
            train_config_path = root / "train.yaml"
            write_yaml(train_config_path, train_config)
            train_sdae(dataset_dir, train_config_path, model_dir)

            cluster_config = copy.deepcopy(read_yaml(Path("configs/hdbscan.yaml")))
            cluster_config["min_cluster_size"] = 3
            cluster_config["min_samples"] = 3
            cluster_config["known_fault_labels"] = ["bearing_impulse"]
            cluster_config_path = root / "hdbscan.yaml"
            write_yaml(cluster_config_path, cluster_config)

            result = build_fault_dictionary(model_dir, dataset_dir, cluster_config_path, dictionary_dir)

            self.assertGreaterEqual(result["dictionary_entry_count"], 1)
            self.assertTrue((dictionary_dir / "dictionary_manifest.yaml").exists())
            self.assertTrue((dictionary_dir / "dictionary.json").exists())
            self.assertTrue((dictionary_dir / "cluster_summary.csv").exists())
            self.assertTrue((dictionary_dir / "cluster_plots" / "latent_clusters.png").exists())
            self.assertTrue((dictionary_dir / "known_novel_decisions.csv").exists())

            manifest = read_yaml(dictionary_dir / "dictionary_manifest.yaml")
            self.assertEqual(manifest["hdbscan"]["method"], "hdbscan.HDBSCAN")
            self.assertEqual(manifest["mahalanobis"]["method"], "scipy.stats.chi2.ppf")
            self.assertGreaterEqual(manifest["dictionary_entry_count"], 1)

            with (dictionary_dir / "dictionary.json").open("r", encoding="utf-8") as handle:
                dictionary = json.load(handle)
            first_entry = dictionary["entries"][0]
            self.assertEqual(first_entry["covariance_estimator"], "sklearn.covariance.LedoitWolf")
            self.assertEqual(first_entry["label"], "bearing_impulse")
            self.assertGreater(first_entry["sample_count"], 0)
            self.assertIn("mahalanobis_chi_square_threshold", first_entry)
            self.assertIn("mahalanobis_effective_threshold", first_entry)
            self.assertIn("mahalanobis_empirical_status", first_entry)
            self.assertLessEqual(
                float(first_entry["mahalanobis_effective_threshold"]),
                float(first_entry["mahalanobis_chi_square_threshold"]),
            )

    def test_cluster_dictionary_decision_uses_passing_cluster_gate(self) -> None:
        import numpy as np

        dictionary = {
            "clustering": {"config": {"cluster_match_min_member_fraction": 0.5}},
            "entries": [
                {
                    "fault_id": "near_but_outside_gate",
                    "label": "near_but_outside_gate",
                    "cluster_label": 0,
                    "centroid": [0.75, 0.0],
                    "precision": [[1.0, 0.0], [0.0, 1.0]],
                    "mahalanobis_threshold": 0.0001,
                },
                {
                    "fault_id": "passing_fault",
                    "label": "passing_fault",
                    "cluster_label": 1,
                    "centroid": [1.0, 0.0],
                    "precision": [[1.0, 0.0], [0.0, 1.0]],
                    "mahalanobis_threshold": 0.25,
                },
            ],
        }
        latents = np.asarray([[0.8, 0.0], [0.9, 0.0], [0.7, 0.0]], dtype=np.float64)

        decision = decide_latent_cluster(latents, dictionary)

        self.assertEqual(decision["decision"], "known")
        self.assertEqual(decision["fault_id"], "passing_fault")
        self.assertEqual(decision["cluster_support_count"], 3)
        self.assertGreaterEqual(decision["cluster_member_inlier_fraction"], 0.5)

    def test_empirical_threshold_rejects_chi_square_only_match(self) -> None:
        import numpy as np

        dictionary = {
            "clustering": {"config": {"cluster_match_min_member_fraction": 0.0}},
            "entries": [
                {
                    "fault_id": "tight_fault",
                    "label": "tight_fault",
                    "cluster_label": 0,
                    "centroid": [0.0, 0.0],
                    "precision": [[1.0, 0.0], [0.0, 1.0]],
                    "mahalanobis_threshold": 5.0,
                    "mahalanobis_effective_threshold": 5.0,
                    "mahalanobis_chi_square_threshold": 32.0,
                }
            ],
        }
        latents = np.asarray([[4.0, 0.0], [4.1, 0.0], [3.9, 0.0]], dtype=np.float64)

        decision = decide_latent_cluster(latents, dictionary)

        self.assertEqual(decision["decision"], "novel_empirical_threshold")
        self.assertEqual(decision["fault_id"], "tight_fault")
        self.assertGreater(decision["distance"], decision["threshold"])
        self.assertLess(decision["distance"], decision["chi_square_threshold"])


if __name__ == "__main__":
    unittest.main()

