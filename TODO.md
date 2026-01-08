# TODO

Refactor ideas discovered during codebase skim:

- Consolidate Basis combination generators into a single parameterized routine to reduce
  duplicated sorting/ordering logic in `src/basis.cpp`.
- Generalize Hubbard models (1D/2D/3D) to a dimension-agnostic helper using `DynamicIndex`
  to reduce duplicated neighbor enumeration in `src/models/hubbard_model.h`.
- Share inner loops between serial and OpenMP matrix element routines to avoid duplication
  in `src/matrix_elements.h`.
- Merge duplicate initialization paths in `IndexedHashSet` (range vs moved vector) into a
  single helper to simplify `src/indexed_hash_set.h`.
- Unify `Expression::add_to_map` overloads via forwarding and centralize repeated multiply
  and append logic in `src/expression.cpp` and `src/expression.h`.
- Clean up `NormalOrderer` API: remove or implement the unused overload and centralize
  consecutive-element checks in `src/normal_order.h` and `src/normal_order.cpp`.
- Cache `DynamicIndex` total size to avoid repeated recomputation in `src/index.h`.
- Add `EXPECT_FALSE` macro to `tests/framework.h` to avoid `EXPECT_TRUE(!...)` usage.
- Add Lanczos, time evolution, etc.
- Add Chebyshev, density of states, etc
- Implement total momentum/relative position hamiltonians for Hubbard Model. 