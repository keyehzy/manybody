# manybody

manybody is a framework for many-body physics.

## Features

- Operator algebra: creation/annihilation operators, terms, normal ordering, commutators, and symbolic expressions.
- Model building blocks: 1D/2D/3D Hubbard models (real-space, momentum, and relative-coordinate variants).
- Numerical routines: Lanczos, KPM projector, Krylov time evolution, Wegner flow, Schrieffer-Wolff, block RG, and optical conductivity utilities.
- OpenMP-enabled and Armadillo-based linear algebra.

## Requirements

- C++20 compiler
- CMake
- Armadillo
- OpenMP
- OpenBLAS
- FFTW3
- ARPACK
- SuperLU

## Build

```bash
cmake -B build -S . -G Ninja -DCMAKE_BUILD_TYPE=Release -DMANYBODY_BUILD_TESTS=ON
cmake --build build
```

## Examples

After building, run any executable under `build/examples/`.

## Tests

Enable tests with `-DMANYBODY_BUILD_TESTS=ON`. Then run:

```bash
./build/tests/tests
```

## Minimal usage example

```cpp
#include <armadillo>
#include <iostream>

#include "algebra/basis.h"
#include "algebra/matrix_elements.h"
#include "algebra/model/hubbard_model.h"

int main() {
  const size_t sites = 4;
  const size_t particles = 2;

  Basis basis = Basis::with_fixed_particle_number(sites, particles);
  HubbardModel model(/*t=*/1.0, /*u=*/4.0, sites);

  arma::cx_mat H = compute_matrix_elements<arma::cx_mat>(basis, model.hamiltonian());
  std::cout << H << "\n";
  return 0;
}
```

## License

MIT. See `LICENSE`.
