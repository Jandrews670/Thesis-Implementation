#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_linux_common.sh"

run_py -m unittest tests.test_objective_7
run_py -m usv_faults.cli attach-data --source cwru --config configs/public_cwru.yaml --out data/raw/public_cwru
run_py -m usv_faults.cli qc --trial data/raw/public_cwru/2026-05-17_CWRU_normal_train_97
run_py -m usv_faults.cli make-dataset --config configs/dataset_public_cwru.yaml --out data/processed/datasets/ds_public_cwru_objective_7
run_py -m usv_faults.cli train-sdae --dataset data/processed/datasets/ds_public_cwru_objective_7 --config configs/baseline_sdae_public_cwru.yaml --out artifacts/models/run_public_cwru_sdae_objective_7
run_py -m usv_faults.cli build-dictionary --model artifacts/models/run_public_cwru_sdae_objective_7 --dataset data/processed/datasets/ds_public_cwru_objective_7 --config configs/hdbscan_public_cwru.yaml --out artifacts/dictionaries/dict_public_cwru_objective_7
run_py -m usv_faults.cli evaluate --model artifacts/models/run_public_cwru_sdae_objective_7 --dictionary artifacts/dictionaries/dict_public_cwru_objective_7 --dataset data/processed/datasets/ds_public_cwru_objective_7 --out runs/reports/objective_7_public_cwru
