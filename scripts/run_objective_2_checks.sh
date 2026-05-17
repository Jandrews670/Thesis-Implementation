#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_linux_common.sh"

run_unit_tests_unless_skipped
run_py -m usv_faults.cli attach-data --source synthetic --config configs/poc_synthetic_smoke.yaml --out data/raw/trials_smoke
run_py -m usv_faults.cli preview --trial data/raw/trials_smoke/2026-05-14_POC_B0_nominal_T001
run_py -m usv_faults.cli make-dataset --config configs/dataset_poc_synthetic_smoke.yaml --out data/processed/datasets/ds_poc_synthetic_smoke

