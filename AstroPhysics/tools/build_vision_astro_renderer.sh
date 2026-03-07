#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUILD_DIR="${PROJECT_DIR}/build-cmake"
TARGET="${1:-vision_astro_interactions_3d_cpp}"

if "${PROJECT_DIR}/cmakew.sh" --version >/dev/null 2>&1; then
  "${PROJECT_DIR}/cmakew.sh" -S "${PROJECT_DIR}" -B "${BUILD_DIR}"
  "${PROJECT_DIR}/cmakew.sh" --build "${BUILD_DIR}" --target "${TARGET}" -j
  echo "Built target '${TARGET}' in ${BUILD_DIR}"
  exit 0
fi

# Fallback path when cmake is not available: use existing native raylib object cache.
if [[ "${TARGET}" != "vision_astro_interactions_3d_cpp" ]]; then
  echo "Error: fallback build only supports target '${TARGET}'." >&2
  exit 2
fi

OUT_DIR="${PROJECT_DIR}/build-native"
OUT_BIN="${OUT_DIR}/${TARGET}"
mkdir -p "${OUT_DIR}" "${BUILD_DIR}"

clang++ -std=c++17 -O2 \
  -I "${PROJECT_DIR}/third_party/raylib/include" \
  "${PROJECT_DIR}/vision/render3d/astro_interactions_renderer_3d.cpp" \
  "${OUT_DIR}/raylib-local/rcore.o" \
  "${OUT_DIR}/raylib-local/rglfw.o" \
  "${OUT_DIR}/raylib-local/rmodels.o" \
  "${OUT_DIR}/raylib-local/rshapes.o" \
  "${OUT_DIR}/raylib-local/rtext.o" \
  "${OUT_DIR}/raylib-local/rtextures.o" \
  -framework Cocoa -framework OpenGL -framework IOKit -framework CoreVideo \
  -o "${OUT_BIN}"

ln -sf "../build-native/${TARGET}" "${BUILD_DIR}/${TARGET}"
echo "Built target '${TARGET}' in ${OUT_DIR} (fallback mode)"
echo "Symlinked ${BUILD_DIR}/${TARGET} -> ../build-native/${TARGET}"
