#!/usr/bin/env bash
set -euo pipefail

if command -v cmake >/dev/null 2>&1; then
  exec "$(command -v cmake)" "$@"
fi

for candidate in \
  "/opt/homebrew/bin/cmake" \
  "/usr/local/bin/cmake" \
  "/Applications/CMake.app/Contents/bin/cmake"
do
  if [[ -x "$candidate" ]]; then
    exec "$candidate" "$@"
  fi
done

echo "Error: cmake not found in PATH or standard macOS locations." >&2
echo "Install CMake (Homebrew: 'brew install cmake') or CMake.app, then rerun." >&2
exit 127
