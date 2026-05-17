from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import savemat

from usv_faults.clustering.fault_dictionary import build_fault_dictionary
from usv_faults.config import read_yaml, write_yaml
from usv_faults.data_sources.cwru import CWRUBearingSource
from usv_faults.evaluation.reports import evaluate_pipeline
from usv_faults.preprocessing.datasets import make_dataset
from usv_faults.storage.trials import quality_check_trial, read_manifest
from usv_faults.training.train_sdae import train_sdae


class ObjectiveSevenTests(unittest.TestCase):
    def test_cwru_adapter_runs_existing_objectives_on_reduced_public_profile(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            mat_dir = root / "mat"
            raw_root = root / "raw"
            dataset_dir = root / "dataset"
            model_dir = root / "model"
            dictionary_dir = root / "dictionary"
            reports_dir = root / "reports"
            mat_dir.mkdir()

            sample_rate_hz = 1000
            _write_fixture_mat(mat_dir / "healthy_train.mat", "X001_DE_time", sample_rate_hz, "healthy", seed=1)
            _write_fixture_mat(mat_dir / "healthy_validation.mat", "X002_DE_time", sample_rate_hz, "healthy", seed=2)
            _write_fixture_mat(mat_dir / "inner.mat", "X003_DE_time", sample_rate_hz, "inner", seed=3)
            _write_fixture_mat(mat_dir / "ball.mat", "X004_DE_time", sample_rate_hz, "ball", seed=4)

            source_config_path = root / "public_cwru_fixture.yaml"
            write_yaml(source_config_path, _source_config(mat_dir, sample_rate_hz))
            created = CWRUBearingSource.from_config_path(source_config_path).attach(raw_root)
            self.assertEqual(len(created), 4)

            report = quality_check_trial(raw_root / "fixture_cwru_normal_train")
            manifest = read_manifest(raw_root / "fixture_cwru_normal_train")
            self.assertIn(report.status, {"accepted", "accepted_with_notes"})
            self.assertEqual(manifest.collection.source_type, "external_cwru")
            self.assertEqual(manifest.sensor_config.current_channels, [])
            self.assertEqual(manifest.sensor_config.vibration_channels, ["drive_end_vibration"])
            self.assertEqual(manifest.collection.model_extra["channel_profile"]["expected_input_dim"], 100)

            dataset_config_path = root / "dataset.yaml"
            write_yaml(dataset_config_path, _dataset_config(raw_root))
            dataset = make_dataset(dataset_config_path, dataset_dir)
            self.assertEqual(dataset["input_dim"], 100)
            self.assertEqual(dataset["window_count"], 80)

            train_config_path = root / "train.yaml"
            write_yaml(train_config_path, _train_config())
            train_sdae(dataset_dir, train_config_path, model_dir)

            dictionary_config_path = root / "hdbscan.yaml"
            write_yaml(dictionary_config_path, _dictionary_config())
            dictionary = build_fault_dictionary(model_dir, dataset_dir, dictionary_config_path, dictionary_dir)
            self.assertGreaterEqual(dictionary["dictionary_entry_count"], 1)

            evaluation = evaluate_pipeline(model_dir, dictionary_dir, dataset_dir, reports_dir)
            self.assertEqual(evaluation["window_count"], 80)
            self.assertTrue((reports_dir / "poc_detection_metrics.csv").exists())
            self.assertTrue((reports_dir / "poc_isolation_metrics.csv").exists())
            self.assertTrue((reports_dir / "poc_performance_metrics.csv").exists())

            dataset_manifest = read_yaml(dataset_dir / "dataset_manifest.yaml")
            self.assertEqual(dataset_manifest["source_type"], "external_cwru")
            self.assertEqual(dataset_manifest["preprocessing"]["expected_input_dim"], 100)
            decisions = pd.read_csv(reports_dir / "poc_window_decisions.csv")
            fault_decisions = decisions[decisions["is_fault"].astype(bool)]
            self.assertGreater(int(fault_decisions["is_anomaly"].sum()), 0)
            self.assertIn("runtime_cluster_label", decisions.columns)
            self.assertIn("cluster_member_inlier_fraction", decisions.columns)


def _write_fixture_mat(path: Path, variable: str, sample_rate_hz: int, kind: str, seed: int) -> None:
    rng = np.random.default_rng(seed)
    sample_count = int(2.0 * sample_rate_hz)
    t = np.arange(sample_count, dtype=np.float64) / sample_rate_hz
    signal = 0.25 * np.sin(2.0 * np.pi * 35.0 * t)
    signal += 0.04 * rng.standard_normal(sample_count)
    if kind == "inner":
        impulses = ((t * 90.0) % 1.0) < 0.05
        signal[impulses] += 2.0
        signal += 0.6 * np.sin(2.0 * np.pi * 140.0 * t)
    elif kind == "ball":
        signal += 1.1 * np.sin(2.0 * np.pi * 75.0 * t) * (1.0 + 0.4 * np.sin(2.0 * np.pi * 5.0 * t))
    savemat(path, {variable: signal.reshape(-1, 1)})


def _source_config(mat_dir: Path, sample_rate_hz: int) -> dict:
    return {
        "attachment_id": "fixture_public_cwru",
        "source_type": "external_cwru",
        "schema_version": "0.1.0",
        "source_url": "https://engineering.case.edu/bearingdatacenter/download-data-file",
        "sampling": {"raw_sample_rate_hz": sample_rate_hz},
        "channel_profile": {
            "name": "fixture_cwru_vibration_only",
            "expected_input_dim": 100,
            "vibration_channels": ["drive_end_vibration"],
            "current_channels": [],
            "scalar_channels": [],
            "scalar_features": [],
        },
        "trials": [
            {
                "trial_id": "fixture_cwru_normal_train",
                "path": str(mat_dir / "healthy_train.mat"),
                "file_name": "healthy_train.mat",
                "mat_variable": "X001_DE_time",
                "duration_s": 2.0,
                "baseline_id": 0,
                "baseline_name": "fixture_normal",
                "fault_label": "none",
                "fault_induced": False,
            },
            {
                "trial_id": "fixture_cwru_normal_validation",
                "path": str(mat_dir / "healthy_validation.mat"),
                "file_name": "healthy_validation.mat",
                "mat_variable": "X002_DE_time",
                "duration_s": 2.0,
                "baseline_id": 0,
                "baseline_name": "fixture_normal",
                "fault_label": "none",
                "fault_induced": False,
            },
            {
                "trial_id": "fixture_cwru_inner_fault",
                "path": str(mat_dir / "inner.mat"),
                "file_name": "inner.mat",
                "mat_variable": "X003_DE_time",
                "duration_s": 2.0,
                "baseline_id": 0,
                "baseline_name": "fixture_fault",
                "fault_label": "inner_race_fault_007",
            },
            {
                "trial_id": "fixture_cwru_ball_fault",
                "path": str(mat_dir / "ball.mat"),
                "file_name": "ball.mat",
                "mat_variable": "X004_DE_time",
                "duration_s": 2.0,
                "baseline_id": 0,
                "baseline_name": "fixture_fault",
                "fault_label": "ball_fault_007",
            },
        ],
    }


def _dataset_config(raw_root: Path) -> dict:
    return {
        "dataset_id": "fixture_public_cwru",
        "source_type": "external_cwru",
        "raw_trial_root": str(raw_root),
        "source_trials": [
            "fixture_cwru_normal_train",
            "fixture_cwru_normal_validation",
            "fixture_cwru_inner_fault",
            "fixture_cwru_ball_fault",
        ],
        "windowing": {"window_ms": 100, "stride_ms": 100},
        "preprocessing": {
            "vibration_sample_rate_hz": 1000,
            "current_sample_rate_hz": 1000,
            "scalar_features": [],
            "expected_input_dim": 100,
        },
        "scaling": {
            "method": "standard",
            "fit_on": "healthy_train_only",
            "channel_profile": "fixture_cwru_vibration_only",
        },
        "split": {
            "strategy": "by_trial",
            "train": ["fixture_cwru_normal_train"],
            "validation": ["fixture_cwru_normal_validation"],
            "test": ["fixture_cwru_inner_fault", "fixture_cwru_ball_fault"],
        },
    }


def _train_config() -> dict:
    return {
        "model": {
            "input_dim": 100,
            "hidden_dims": [24],
            "latent_dim": 4,
            "hidden_activation": "relu",
            "output_activation": "sigmoid",
            "masking_noise": 0.05,
            "l1_lambda": 0.0001,
        },
        "training": {
            "optimizer": "adam",
            "learning_rate": 0.001,
            "batch_size": 8,
            "epochs": 2,
            "early_stopping": False,
            "seed": 20260517,
        },
        "threshold": {"method": "validation_percentile", "target_false_positive_rate": 0.02},
    }


def _dictionary_config() -> dict:
    return {
        "rolling_window_size": 300,
        "min_cluster_size": 2,
        "min_samples": 1,
        "metric": "euclidean",
        "cluster_selection_method": "eom",
        "allow_single_cluster": True,
        "mahalanobis_confidence": 0.99,
        "dictionary_baseline_id": 0,
        "known_fault_labels": ["inner_race_fault_007", "ball_fault_007"],
        "withheld_fault_labels": [],
    }


if __name__ == "__main__":
    unittest.main()
