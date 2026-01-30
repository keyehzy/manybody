# AGENTS.md

## Build System

- We use CMake as the build system.
- Use scripts/make.sh to build and test

## Commit

- Format files using script/format.sh before commiting

Example COMMIT:
```
Refactor user service to reduce duplication

Extract shared validation logic into a helper module,
simplifying maintenance and improving test coverage.
```
## Coding Conventions

1. Header Guards: #pragma once
2. Type Aliases: Prefer aliases at class/struct level (using)
3. Constructors: Explicit for conversions; defaulted = default/delete for special members
4. Move Semantics: noexcept on move constructors and helper functions
5. Binary Operators: Free functions using copy-and-modify idiom
6. Includes: Related headers with quotes, standard library headers with angle brackets
7. Naming:
  - Classes/Structs: PascalCase (Expression, Basis, NormalOrderer)
  - Functions: snake_case (compute_matrix_elements)
  - Template parameters: PascalCase (MatrixType, VectorType)
  - Member variables: snake_case with {} default initialization
8. Member Access: Public members for simple data containers
9. Parallelization: OpenMP with critical sections for shared writes
10. Constants: Use immutable/const where appropriate