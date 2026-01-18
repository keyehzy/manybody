#!/bin/bash
set -e

BUILD_DIR="build"
OPENBLAS_ROOT="$(brew --prefix openblas)"

DEBUG_FLAGS="-Wall \
-Wextra \
-Wpedantic \
-Wconversion \
-Wsign-conversion \
-Wdouble-promotion \
-Wshadow \
-Wswitch-enum \
-Wswitch-default \
-Wimplicit-fallthrough \
-Wnon-virtual-dtor \
-Woverloaded-virtual \
-Wdelete-non-virtual-dtor \
-Wformat \
-Wformat-security \
-Werror=format-security \
-Wstrict-aliasing=2 \
-Wnull-dereference \
-Wduplicated-cond \
-Wduplicated-branches \
-Wlogical-op \
-Wold-style-cast \
-Weffc++ \
-Wuseless-cast \
-Wredundant-move \
-Walloc-zero \
-Walloca \
-Wstringop-overflow=4 \
-Wstringop-truncation"

cmake -B "$BUILD_DIR" -S . -G Ninja \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_CXX_COMPILER="/opt/homebrew/Cellar/gcc/15.2.0/bin/g++-15" \
  -DCMAKE_CXX_FLAGS="$DEBUG_FLAGS" \
  -DOpenBLAS_ROOT="${OPENBLAS_ROOT}"

cmake --build "$BUILD_DIR" --verbose
