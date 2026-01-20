#!/bin/bash
set -e

BUILD_DIR="build_sanitize"
OPENBLAS_ROOT="$(brew --prefix openblas)"
LLVM_ROOT="$(brew --prefix llvm)"
LIBOMP_ROOT="$(brew --prefix libomp)"
LIBCXX_ROOT="${LLVM_ROOT}/lib/c++"

SANITIZER_FLAGS="-g -O1 \
-fno-omit-frame-pointer \
-fsanitize=address,undefined \
-fno-sanitize-recover=all"

cmake -B "$BUILD_DIR" -S . -G Ninja \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_CXX_COMPILER="${LLVM_ROOT}/bin/clang++" \
  -DCMAKE_CXX_FLAGS="-Wall -Wextra -Wpedantic $SANITIZER_FLAGS" \
  -DCMAKE_EXE_LINKER_FLAGS="$SANITIZER_FLAGS -L${LIBCXX_ROOT} -Wl,-rpath,${LIBCXX_ROOT} -lc++ -lc++abi" \
  -DOpenBLAS_ROOT="${OPENBLAS_ROOT}" \
  -DOpenMP_ROOT="${LIBOMP_ROOT}"

cmake --build "$BUILD_DIR"

"$BUILD_DIR/tests/tests"
