#!/bin/bash
set -e

BUILD_DIR="build"
OPENBLAS_ROOT="$(brew --prefix openblas)"

cmake -B "$BUILD_DIR" -S . -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DCMAKE_CXX_COMPILER="/opt/homebrew/Cellar/gcc/15.2.0/bin/g++-15" \
  -DCMAKE_CXX_FLAGS="-Wall -Wextra" \
  -DOpenBLAS_ROOT="${OPENBLAS_ROOT}" \
  -DMANYBODY_BUILD_TESTS=ON

cmake --build "$BUILD_DIR" --verbose

"$BUILD_DIR/tests/tests"
