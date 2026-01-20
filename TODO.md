# TODO

Refactor ideas discovered during codebase skim:

- Consolidate Basis combination generators into a single parameterized routine to reduce
  duplicated sorting/ordering logic in `src/basis.cpp`.
- Generalize Hubbard models (1D/2D/3D) to a dimension-agnostic helper using `DynamicIndex`
  to reduce duplicated neighbor enumeration in `src/algebra/hubbard_model.h`.
- Share inner loops between serial and OpenMP matrix element routines to avoid duplication
  in `src/matrix_elements.h`.
- Merge duplicate initialization paths in `IndexedHashSet` (range vs moved vector) into a
  single helper to simplify `src/indexed_hash_set.h`.
- Unify `Expression::add_to_map` overloads via forwarding and centralize repeated multiply
  and append logic in `src/expression.cpp` and `src/expression.h`.
- Clean up `NormalOrderer` API: remove or implement the unused overload and centralize
  consecutive-element checks in `src/normal_order.h` and `src/normal_order.cpp`.
- Cache `DynamicIndex` total size to avoid repeated recomputation in `src/index.h`.
- Add small guard/test for edge-case `p_dim` values in block Wegner flow API.
- Add Lanczos, time evolution, etc.
- Add Chebyshev, density of states, etc
- Implement total momentum/relative position hamiltonians for Hubbard Model. 
- LinearOperator perf: avoid temporaries by adding in-place apply(out, in) and use in Sum/Difference/Composed.
- LinearOperator perf: store operands by reference/pointer or use forwarding to avoid heavy operator copies in compositions.
- LinearOperator perf: consider non-virtual/expression-template path for hot loops to reduce dispatch and allocations.
- LinearOperator perf: reduce modulo cost in Hubbard relative kinetic operator (edge/inner loop or precomputed neighbors).
- Performance: the generic path uses per-site coordinate extraction and offset vectors, which is a bit heavier than hard-coded loops. For the examples, it should be fine; if performance
  matters, you can reuse preallocated coords/offsets vectors inside the loop to avoid allocations.
- Remove trivial wrapper `diagonal_part` around `arma::diagmat(H.diag())` in `src/algorithms/wegner_flow.cpp`.
- Remove trivial wrapper `pair_annihilation` around `pair_creation(r).adjoint()` in `src/algebra/hubbard_model_relative.h`.
- Expression term-size cap: `src/algebra/expression.cpp` hard-codes `12` in `add_to_map` and in
  `operator*=` loops. Replace with a single `constexpr` derived from `Term::static_vector_size`
  (or `Expression::container_type` capacity) so the limit cannot drift if term sizing changes.
- Vector elements map lookup: in `src/algebra/matrix_elements.h`, `product.hashmap[{}]` inserts a
  zero term when the empty operator string is missing. Swap to `find` and default to zero to avoid
  mutating `hashmap` and doing extra allocations in both serial and OpenMP paths.
- OpenMP column writes: `src/algebra/matrix_elements.h` uses `#pragma omp critical` even though each
  thread owns a unique `j` column (`#pragma omp for` over `j`). Remove the critical section (writes
  are column-exclusive) to avoid unnecessary locking; optionally guard `<omp.h>` and pragmas with
  `_OPENMP` so non-OpenMP builds do not require the header.
- Momentum canonicalization bounds: `src/utils/canonicalize_momentum.h` assumes `momentum.size() ==
  size.size()`. Add a size check (throw `std::invalid_argument` or `assert` + documented precondition)
  before indexing `size[d]` to avoid out-of-bounds access.
- Release-build safety: several constructors/functions rely on `assert` for input validation (e.g.
  `Basis::with_fixed_particle_number*` in `src/algebra/basis.cpp`, `HubbardRelative*` operators in
  `src/numerics/hubbard_relative_operators.h`, `HubbardRelative3*` in
  `src/numerics/hubbard_relative3_operators.h`). Add runtime checks that throw with clear messages
  so misuse fails fast in release builds.
- `static_vector::at` semantics: `src/utils/static_vector.h` defines `at()` but only asserts.
  Either implement bounds-checked `at()` that throws `std::out_of_range` (and include `<stdexcept>`)
  or rename to avoid promising standard `at()` semantics.
- `HubbardModelMomentum` performance: `src/algebra/model/hubbard_model_momentum.h` repeatedly calls
  `index(k)` in the O(N^3) interaction loop. Precompute `std::vector<Index::container_type> coords`
  once (`coords[k] = index(k)`) and reuse for `k1`, `k2`, `q` to reduce allocations and indexing work.
- Relative position basis (direct construction): Create a new `Basis` factory method that directly
  generates relative position states as `Expression` objects. Each basis vector would store the
  Fourier superposition of momentum-space terms with phase coefficients, rather than using a
  transformation matrix approach. This would allow working directly in the relative coordinate
  representation without needing to transform Hamiltonians.
- Investigate HubbardRelative/HubbardModelRelative eigenvalue discrepancy: The `HubbardRelative`
  (LinearOperator) and `HubbardModelRelative` (symbolic) models give different eigenvalues than
  `HubbardModelMomentum` for the same 2-particle system with fixed total momentum. The effective
  model uses `t_eff = -2t*cos(K/2)` with uniform nearest-neighbor hopping, while the standard
  Hubbard model transformed to relative coordinates has a different kinetic structure. Determine
  which formulation correctly represents the 2-particle Hubbard physics.
