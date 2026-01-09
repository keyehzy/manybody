#!/usr/bin/env python3
import os
import subprocess
import sys
import glob
import argparse
import shlex

CXX_COMPILER = "/opt/homebrew/opt/llvm/bin/clang++"
CXX_FLAGS = ["-std=c++20", "-O2", "-Wall", "-Wextra"]
AR = "ar"

BUILD_DIR = "build"
OBJ_DIR = os.path.join(BUILD_DIR, "obj")

def pkg_config(package, flag):
    cmd = ['pkg-config', flag, package]
    output = subprocess.check_output(cmd).decode('utf-8').strip()
    return shlex.split(output)

class OpenMP:
    Cflags = [""]
    Libs = ["-L/opt/homebrew/Cellar/libomp/21.1.7/lib", "-fopenmp"]

class Armadillo:
    Cflags = pkg_config('armadillo', '--cflags')
    Libs = pkg_config('armadillo', '--libs')

class Target:
    def __init__(
        self,
        name,
        kind,
        sources,
        includes=None,
        libraries=None,
        flags=None,
        deps=None,
        extra_deps=None,
    ):
        self.name = name
        self.kind = kind
        self.sources = sources
        self.includes = includes or []
        self.libraries = libraries or []
        self.flags = flags or []
        self.deps = deps or []
        self.extra_deps = extra_deps or []

    @property
    def output(self):
        if self.kind == "library":
            return os.path.join(BUILD_DIR, f"lib{self.name}.a")
        return os.path.join(BUILD_DIR, self.name)

    def object_path(self, source):
        return os.path.join(OBJ_DIR, self.name, source + ".o")

    def compile_objects(self):
        print("[debug] compile objects")
        print("[debug] includes: ", self.includes)
        print("[debug] libraries: ", self.libraries)
        print("[debug] flags: ", self.flags)
        includes = ["-I" + include for include in self.includes]
        object_files = []
        for source in self.sources:
            obj = self.object_path(source)
            object_files.append(obj)
            if should_rebuild(obj, [source, *self.extra_deps]):
                print(f"[compile] {self.name}: {source} -> {obj}")
                os.makedirs(os.path.dirname(obj), exist_ok=True)
                cmd = [CXX_COMPILER, *CXX_FLAGS, *self.flags, *includes, "-c", source, "-o", obj, *self.libraries]
                print("[debug] ", ' '.join(cmd))
                subprocess.run(cmd, check=True)
            else:
                print(f"[compile] {self.name}: {source} up to date")
        return object_files

    def build(self):
        print("[debug] build")
        print(f"[target] {self.name} ({self.kind})")
        object_files = self.compile_objects()
        if self.kind == "library":
            if should_rebuild(self.output, object_files):
                print(f"[archive] {self.output}")
                os.makedirs(os.path.dirname(self.output), exist_ok=True)
                cmd = [AR, "rcs", self.output, *object_files]
                subprocess.run(cmd, check=True)
            else:
                print(f"[archive] {self.output} up to date")
        elif self.kind == "executable":
            link_inputs = [*object_files, *(dep.output for dep in self.deps)]
            if should_rebuild(self.output, link_inputs):
                print(f"[link] {self.output}")
                os.makedirs(os.path.dirname(self.output), exist_ok=True)
                includes = ["-I" + include for include in self.includes]
                cmd = [
                    CXX_COMPILER,
                    *CXX_FLAGS,
                    *self.flags,
                    *includes,
                    "-o",
                    self.output,
                    *link_inputs,
                    *self.libraries,
                ]
                print("[debug] ", ' '.join(cmd))
                subprocess.run(cmd, check=True)
            else:
                print(f"[link] {self.output} up to date")
        else:
            raise ValueError(f"Unknown target kind: {self.kind}")


def should_rebuild(output_file, inputs):
    if not os.path.exists(output_file):
        return True

    output_mtime = os.path.getmtime(output_file)
    for path in inputs:
        if os.path.exists(path) and os.path.getmtime(path) > output_mtime:
            return True
    return False


def topo_sort(targets):
    ordered = []
    visiting = set()
    visited = set()

    def visit(target):
        if target.name in visited:
            return
        if target.name in visiting:
            raise RuntimeError(f"Cycle detected at {target.name}")
        visiting.add(target.name)
        for dep in target.deps:
            visit(dep)
        visiting.remove(target.name)
        visited.add(target.name)
        ordered.append(target)

    for target in targets:
        visit(target)
    return ordered


def build_all():
    if not os.path.exists(BUILD_DIR):
        os.makedirs(BUILD_DIR)
    print("[build] start")
    for target in topo_sort(TARGETS):
        target.build()
    print("[build] done")


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


def filter_files(files):
    return sorted({file for file in files if not file.endswith("~")})


def find_sources(directory, exclude_files=None):
    exclude_files = set(exclude_files or [])
    sources = filter_files(glob.glob(os.path.join(directory, "*.cpp")))
    sources += filter_files(glob.glob(os.path.join(directory, "**", "*.cpp"), recursive=True))
    return [source for source in sources if source not in exclude_files]


def find_headers(directories):
    patterns = []
    for directory in directories:
        patterns.extend(
            [
                os.path.join(directory, "*.h"),
                os.path.join(directory, "*.hpp"),
                os.path.join(directory, "*.hh"),
            ]
        )
    headers = []
    for pattern in patterns:
        headers.extend(glob.glob(pattern))
    return filter_files(headers)


def find_test_deps():
    patterns = [
        "tests/*.cpp",
        "tests/*.h",
        "src/*.cpp",
        "src/*.h",
    ]
    deps = []
    for pattern in patterns:
        deps.extend(glob.glob(pattern))
    return filter_files(deps)


def format_code():
    for file in find_cpp_files():
        print(f"Formatting {file}")
        subprocess.run(['clang-format', '-i', file], check=True)


manybody = Target(
    name="manybody",
    kind="library",
    sources=find_sources("src", exclude_files=["src/main.cpp"]),
    includes=[
        "src",
        "third-party",
    ],
    flags=[*Armadillo.Cflags, *OpenMP.Cflags],
    libraries=[*OpenMP.Libs],
    extra_deps=find_headers(["src"]),
)

tests = Target(
    name="tests",
    kind="executable",
    sources=["tests/main.cpp"],
    includes=[
        "src",
        "tests",
        "third-party",
    ],
    flags =[*Armadillo.Cflags],
    libraries=[*Armadillo.Libs, *OpenMP.Libs],
    deps=[manybody],
    extra_deps=find_test_deps(),
)

def example_target(name, source):
    return Target(
        name=name,
        kind="executable",
        sources=[source],
        includes=[
            "src",
            "third-party",
        ],
        flags =[*Armadillo.Cflags],
        libraries=[*Armadillo.Libs, *OpenMP.Libs],
        deps=[manybody],
    )


app = example_target(
    "main",
    "src/main.cpp",
)

example_hubbard_ed = example_target(
    "example_hubbard_ed",
    "examples/hubbard_exact_diagonalization.cpp",
)

example_hubbard_3d_sparse = example_target(
    "example_hubbard_3d_sparse",
    "examples/hubbard_3d_sparse_lowest.cpp",
)

example_hubbard_1d_schriffer_wolff = example_target(
    "example_hubbard_1d_schriffer_wolff",
    "examples/hubbard_1d_schriffer_wolff.cpp",
)

example_hubbard_1d_schriffer_wolff_truncated = example_target(
    "example_hubbard_1d_schriffer_wolff_truncated",
    "examples/hubbard_1d_schriffer_wolff_truncated.cpp",
)

example_hubbard_1d_schriffer_wolff_benchmark = example_target(
    "example_hubbard_1d_schriffer_wolff_benchmark",
    "examples/hubbard_1d_schriffer_wolff_benchmark.cpp",
)

example_tight_binding_fourier = example_target(
    "example_tight_binding_fourier",
    "examples/tight_binding_fourier.cpp",
)

example_hubbard_relative_linear_operator = example_target(
    "example_hubbard_relative_linear_operator",
    "examples/hubbard_relative_linear_operator.cpp",
)

TARGETS = [
    manybody,
    app,
    example_hubbard_ed,
    example_hubbard_3d_sparse,
    example_hubbard_1d_schriffer_wolff,
    example_hubbard_1d_schriffer_wolff_truncated,
    example_hubbard_1d_schriffer_wolff_benchmark,
    example_tight_binding_fourier,
    example_hubbard_relative_linear_operator,
    tests,
]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build system for the project")
    parser.add_argument('--format', action='store_true', help='Run clang-format on all C++ source files')
    parser.add_argument('--flags', type=str, help='Additional compiler flags')
    parser.add_argument('--includes', type=str, help='Additional INCLUDES')
    parser.add_argument('--libs', type=str, help='Additional LIBRARIES')
    args = parser.parse_args()

    if args.flags:
        CXX_FLAGS.extend(args.flags.split())

    if args.includes:
        for target in TARGETS:
            target.includes.extend(args.includes.split())

    if args.libs:
        for target in TARGETS:
            target.libraries.extend(args.libs.split())

    try:
        if args.format:
            format_code()

        build_all()

    except subprocess.CalledProcessError as e:
        sys.exit(1)
