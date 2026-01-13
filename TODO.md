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
- Add `EXPECT_FALSE` macro to `tests/framework.h` to avoid `EXPECT_TRUE(!...)` usage.
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
