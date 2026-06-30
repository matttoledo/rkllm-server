#!/usr/bin/env bash
# Build the rkllm_server binary natively on an aarch64 host (Orange Pi 5,
# Radxa ROCK 5, etc). Cross-compile from x86 is not handled here.
#
# Override the rknn-llm runtime path with:
#   RKLLM_RUNTIME_DIR=/path/to/librkllm_api ./build.sh
set -euo pipefail

SRC="$(cd "$(dirname "$0")" && pwd)"
BUILD="$SRC/build"
mkdir -p "$BUILD"
cd "$BUILD"

CMAKE_ARGS=(-DCMAKE_BUILD_TYPE=Release)
if [[ -n "${RKLLM_RUNTIME_DIR:-}" ]]; then
    CMAKE_ARGS+=(-DRKLLM_RUNTIME_DIR="$RKLLM_RUNTIME_DIR")
fi

cmake "${CMAKE_ARGS[@]}" "$SRC"
make -j"$(nproc)"

echo
echo "✓ Built: $BUILD/rkllm_server"
file "$BUILD/rkllm_server"
