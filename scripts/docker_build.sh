#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

PLATFORM="${1:-}"
TAG="${TAG:-usv-faults:dev}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cpu}"

if [[ -n "${PLATFORM}" ]]; then
  docker buildx build --platform "${PLATFORM}" --load -t "${TAG}" --build-arg "TORCH_INDEX_URL=${TORCH_INDEX_URL}" .
else
  docker compose build --build-arg "TORCH_INDEX_URL=${TORCH_INDEX_URL}"
fi
