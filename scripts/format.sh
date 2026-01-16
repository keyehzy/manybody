#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(git rev-parse --show-toplevel)"

find "$ROOT_DIR" \
  -type d -name third-party -prune -o \
  -type f \( \
    -name '*.c' -o \
    -name '*.cc' -o \
    -name '*.cpp' -o \
    -name '*.cxx' -o \
    -name '*.h' -o \
    -name '*.hh' -o \
    -name '*.hpp' -o \
    -name '*.hxx' \
  \) -print0 \
  | xargs -0 clang-format -i
