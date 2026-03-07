#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUILD_DIR="${PROJECT_DIR}/build-cmake"
TARGET="${1:-vision_astro_interactions_3d_cpp}"

"${PROJECT_DIR}/cmakew.sh" -S "${PROJECT_DIR}" -B "${BUILD_DIR}"
"${PROJECT_DIR}/cmakew.sh" --build "${BUILD_DIR}" --target "${TARGET}" -j

echo "Built target '${TARGET}' in ${BUILD_DIR}"
