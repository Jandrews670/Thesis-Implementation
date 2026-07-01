from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from usv_faults.clustering.fault_dictionary import build_fault_dictionary
from usv_faults.config import load_config, read_yaml, write_yaml
from usv_faults.data_sources.synthetic_usv import SyntheticUSVSource
from usv_faults.evaluation.reports import evaluate_pipeline
from usv_faults.evaluation.trial_runner import run_replay_trial
from usv_faults.preprocessing.datasets import make_dataset
from usv_faults.schemas import SyntheticConfig
from usv_faults.training.train_sdae import train_sdae


class ObjectiveFiveTests(unittest.TestCase):
    def test_evaluate_and_replay_write_milestone_five_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            raw_root = root / "raw"
            dataset_dir = root / "dataset"
            model_dir = root / "model"
            dictionary_dir = root / "dictionary"
            reports_dir = root / "reports"
            logs_dir = root / "logs"

            _build_smoke_artifacts(raw_root, dataset_dir, model_dir, dictionary_dir, root)

            evaluation = evaluate_pipeline(model_dir, dictionary_dir, dataset_dir, reports_dir)
            self.assertEqual(evaluation["window_count"], 60)
            self.assertEqual(evaluation["metric_warmup_windows"], 10)
            self.assertGreater(evaluation["metric_excluded_window_count"], 0)
            self.assertTrue((reports_dir / "poc_detection_metrics.csv").exists())
            self.assertTrue((reports_dir / "poc_isolation_metrics.csv").exists())
            self.assertTrue((reports_dir / "poc_event_metrics.csv").exists())
            self.assertTrue((reports_dir / "poc_event_decisions.csv").exists())
            self.assertTrue((reports_dir / "poc_cross_domain_metrics.csv").exists())
            self.assertTrue((reports_dir / "poc_performance_metrics.csv").exists())
            self.assertTrue((reports_dir / "poc_summary.md").exists())
            self.assertGreater(evaluation["performance_metric_count"], 0)

            detection = pd.read_csv(reports_dir / "poc_detection_metrics.csv")
            isolation = pd.read_csv(reports_dir / "poc_isolation_metrics.csv")
            event_metrics = pd.read_csv(reports_dir / "poc_event_metrics.csv")
            window_decisions = pd.read_csv(reports_dir / "poc_window_decisions.csv")
            event_decisions = pd.read_csv(reports_dir / "poc_event_decisions.csv")
            cross_domain = pd.read_csv(reports_dir / "poc_cross_domain_metrics.csv")
            performance = pd.read_csv(reports_dir / "poc_performance_metrics.csv")
            self.assertIn("false_positive_rate", detection.columns)
            self.assertIn("true_fault_detection_rate", detection.columns)
            self.assertIn("true_fault_isolation_rate", isolation.columns)
            self.assertIn("dbcv_status", isolation.columns)
            self.assertIn("event_true_fault_isolation_rate", event_metrics.columns)
            self.assertIn("event_false_positive_rate", event_metrics.columns)
            self.assertIn("event_fault_detection_rate", event_metrics.columns)
            self.assertIn("metric_excluded", window_decisions.columns)
            self.assertIn("state_window_index", window_decisions.columns)
            self.assertIn("event_decision", event_decisions.columns)
            self.assertIn("event_anomaly_fraction", event_decisions.columns)
            self.assertIn("metric_excluded", event_decisions.columns)
            metric_excluded = window_decisions["metric_excluded"].astype(str).str.lower().eq("true")
            self.assertEqual(evaluation["metric_excluded_window_count"], int(metric_excluded.sum()))
            self.assertLess(
                int(
                    detection.loc[
                        detection["scope"].eq("overall"),
                        "window_count",
                    ].iloc[0]
                ),
                evaluation["window_count"],
            )
            self.assertEqual(
                int(
                    detection.loc[
                        detection["scope"].eq("overall"),
                        "window_count",
                    ].iloc[0]
                ),
                evaluation["metric_window_count"],
            )
            self.assertEqual(set(cross_domain["baseline_id"]), {1, 2, 3, 4})
            performance_metrics = set(performance["metric"])
            self.assertIn("estimated_forward_linear_flops_per_window", performance_metrics)
            self.assertIn("estimated_training_linear_flops_total", performance_metrics)
            self.assertIn("training_cpu_usage_percent_all_cores", performance_metrics)
            self.assertIn("training_peak_ram_mb", performance_metrics)
            self.assertIn("inference_cpu_usage_percent_all_cores", performance_metrics)
            self.assertIn("inference_peak_ram_mb", performance_metrics)
            self.assertGreater(
                float(performance.loc[performance["metric"].eq("estimated_forward_linear_flops_per_window"), "value"].iloc[0]),
                0.0,
            )
            self.assertGreater(
                float(performance.loc[performance["metric"].eq("estimated_training_linear_flops_total"), "value"].iloc[0]),
                0.0,
            )
            self.assertGreater(
                float(performance.loc[performance["metric"].eq("training_peak_ram_mb"), "value"].iloc[0]),
                0.0,
            )
            self.assertGreater(
                float(performance.loc[performance["metric"].eq("inference_peak_ram_mb"), "value"].iloc[0]),
                0.0,
            )

            replay = run_replay_trial(
                "replay",
                raw_root / "2026-05-14_POC_B0_fault_bearing_T001",
                model_dir,
                dictionary_dir,
                logs_dir,
            )
            self.assertGreater(replay["anomaly_count"], 0)
            replay_log = Path(replay["out_path"])
            self.assertTrue(replay_log.exists())
            decisions = pd.read_csv(replay_log)
            self.assertEqual(
                list(decisions.columns),
                [
                    "timestamp_s",
                    "trial_id",
                    "reconstruction_error",
                    "threshold",
                    "is_anomaly",
                    "cluster_label",
                    "dictionary_decision",
                    "decision_basis",
                    "matched_fault_id",
                    "matched_fault_label",
                    "mahalanobis_distance_sq",
                    "mahalanobis_threshold",
                    "cluster_support_count",
                    "cluster_member_inlier_fraction",
                ],
            )
            self.assertIn("known", set(decisions["dictionary_decision"]))
            known = decisions[decisions["dictionary_decision"].eq("known")]
            self.assertGreater(float(known["cluster_member_inlier_fraction"].max()), 0.0)


def _build_smoke_artifacts(
    raw_root: Path,
    dataset_dir: Path,
    model_dir: Path,
    dictionary_dir: Path,
    root: Path,
) -> None:
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
    cluster_config["min_samples"] = 1
    cluster_config["min_runtime_cluster_size"] = 3
    cluster_config["known_fault_labels"] = ["bearing_impulse"]
    cluster_config_path = root / "hdbscan.yaml"
    write_yaml(cluster_config_path, cluster_config)
    build_fault_dictionary(model_dir, dataset_dir, cluster_config_path, dictionary_dir)


if __name__ == "__main__":
    unittest.main()

