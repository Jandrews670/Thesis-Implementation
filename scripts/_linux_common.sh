#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

export PYTHONDONTWRITEBYTECODE="${PYTHONDONTWRITEBYTECODE:-1}"
export PYTHONPATH="${PROJECT_ROOT}/src"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/usv_faults_matplotlib_cache}"
mkdir -p "${MPLCONFIGDIR}"

PYTHON="${PYTHON:-python}"

run_py() {
  "${PYTHON}" "$@"
}

run_unit_tests_unless_skipped() {
  if [[ "${SKIP_UNIT_TESTS:-0}" != "1" ]]; then
    run_py -m unittest discover -s tests
  fi
}

