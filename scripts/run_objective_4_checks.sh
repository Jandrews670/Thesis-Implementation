#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_linux_common.sh"

run_unit_tests_unless_skipped
run_py -m usv_faults.cli attach-data --source synthetic --config configs/poc_synthetic_training_smoke.yaml --out data/raw/trials_training_smoke
run_py -m usv_faults.cli make-dataset --config configs/dataset_poc_synthetic_training_smoke.yaml --out data/processed/datasets/ds_poc_synthetic_training_smoke
run_py -m usv_faults.cli train-sdae --dataset data/processed/datasets/ds_poc_synthetic_training_smoke --config configs/baseline_sdae_smoke.yaml --out artifacts/models/run_poc_sdae_smoke_objective_4
run_py -m usv_faults.cli build-dictionary --model artifacts/models/run_poc_sdae_smoke_objective_4 --dataset data/processed/datasets/ds_poc_synthetic_training_smoke --config configs/hdbscan.yaml --out artifacts/dictionaries/dict_poc_b0_smoke_objective_4

