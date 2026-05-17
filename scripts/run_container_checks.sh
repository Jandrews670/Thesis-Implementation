#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_linux_common.sh"

run_py -m unittest discover -s tests
run_py -m usv_faults.cli --help

SKIP_UNIT_TESTS=1 bash scripts/run_objective_1_checks.sh
SKIP_UNIT_TESTS=1 bash scripts/run_objective_2_checks.sh
SKIP_UNIT_TESTS=1 bash scripts/run_objective_3_checks.sh
SKIP_UNIT_TESTS=1 bash scripts/run_objective_4_checks.sh
SKIP_UNIT_TESTS=1 bash scripts/run_objective_5_checks.sh

echo "Container checks completed."

