#!/usr/bin/env python3
import os
import subprocess
import sys
import glob
import argparse
from multiprocessing import Pool

CXX_COMPILER = "/opt/homebrew/opt/llvm/bin/clang++"
CXX_FLAGS = ["-std=c++20", "-O2", "-Wall", "-Wextra"]

INCLUDES = [ "src", "third-party", "/opt/homebrew/Cellar/armadillo/15.2.2/include", "/opt/homebrew/Cellar/libomp/21.1.7/include" ]
LIBRARIES = [ "/opt/homebrew/Cellar/armadillo/15.2.2/lib", "/opt/homebrew/Cellar/libomp/21.1.7/lib" ]

EXTRA_FLAGS = [ "-larmadillo", "-fopenmp" ]

TESTS = [ "tests/operator.cpp", "tests/static_vector.cpp", "tests/term.cpp",
          "tests/expression.cpp", "tests/normal_order.cpp", "tests/basis_test.cpp",
          "tests/indexed_hash_set.cpp", "tests/matrix_elements.cpp", "src/basis.cpp",
          "src/basis.h", "src/indexed_hash_set.h", "src/matrix_elements.h" ]

TARGETS = [
    (["src/main.cpp", "src/expression.cpp"], [ "src/operator.h", "src/term.h", "src/expression.h" ], "build/main"),
    (["tests/main.cpp", "src/expression.cpp", "src/normal_order.cpp"],
     [ "tests/framework.h", *TESTS ], "build/tests"),
]

BUILD_DIR = "build"

def build_target(source_files, dependencies, output_file):
    if should_rebuild(source_files, dependencies, output_file):
        print(f"Building {output_file}...")
        includes = ["-I" + include for include in INCLUDES]
        libraries = ["-L" + library for library in LIBRARIES]
        cmd = [CXX_COMPILER, *CXX_FLAGS, *includes, "-o", output_file, *source_files, *libraries, *EXTRA_FLAGS]
        subprocess.run(cmd, check=True)
        print(f"Successfully built {output_file}")
        return True
    else:
        print(f"{output_file} is up to date.")
        return True


def should_rebuild(source_files, dependencies, output_file):
    if not os.path.exists(output_file):
        return True

    output_mtime = os.path.getmtime(output_file)
    inputs = list(source_files) + dependencies
    for src in inputs:
        if os.path.exists(src) and os.path.getmtime(src) > output_mtime:
            return True
    return False


def build_all():
    if not os.path.exists(BUILD_DIR):
        os.makedirs(BUILD_DIR)
    with Pool(processes=os.cpu_count()) as pool:
        pool.starmap(build_target, TARGETS)


EXCLUDE_DIRS = {"third-party"}


def is_excluded(path):
    parts = os.path.normpath(path).split(os.sep)
    return parts and parts[0] in EXCLUDE_DIRS


def find_cpp_files():
    extensions = ['*.cpp', '*.h', '*.hpp', '*.cc', '*.cxx']
    files = []
    for ext in extensions:
        files.extend(glob.glob(ext))
        files.extend(glob.glob(f'**/{ext}', recursive=True))
    return [file for file in set(files) if not is_excluded(file)]

def format_code():
    for file in find_cpp_files():
        print(f"Formatting {file}")
        subprocess.run(['clang-format', '-i', file], check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build system for the project")
    parser.add_argument('--format', action='store_true', help='Run clang-format on all C++ source files')
    parser.add_argument('--run', action='store_true', help='Run the target')
    parser.add_argument('--flags', type=str, help='Additional compiler flags')
    parser.add_argument('--includes', type=str, help='Additional INCLUDES')
    parser.add_argument('--libs', type=str, help='Additional LIBRARIES')
    args = parser.parse_args()

    if args.flags:
        CXX_FLAGS.extend(args.flags.split())

    if args.includes:
        INCLUDES.extend(args.includes.split())

    if args.libs:
        LIBRARIES.extend(args.libs.split())

    try:
        if args.format:
            format_code()

        build_all()

        if args.run:
            _, _, exe = TARGETS[0]
            subprocess.run([exe], check=True)

    except subprocess.CalledProcessError as e:
        sys.exit(1)
