# AGENTS.md

## Build System

The build system is located in @build.py.

```bash
./build.py --help
usage: build.py [-h] [--format] [--run] [--flags FLAGS] [--includes INCLUDES] [--libs LIBS]

Build system for the project

options:
  -h, --help           show this help message and exit
  --format             Run clang-format on all C++ source files
  --run                Run the target
  --flags FLAGS        Additional compiler flags
  --includes INCLUDES  Additional INCLUDES
  --libs LIBS          Additional LIBRARIES

Compilation finished at Thu Jan  1 17:59:12, duration 0.77 s
```

## Commit message

Use the following git commit template:

```
<Short summary in imperative mood, ≤ 50 chars>

<Optional body wrapped at ~72 chars explaining:
- what changed
- why it changed
- side effects or constraints>
```

Example:


```
Refactor user service to reduce duplication

Extract shared validation logic into a helper module,
simplifying maintenance and improving test coverage.
```