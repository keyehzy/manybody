# TODO

Refactor ideas discovered during codebase skim:

- Consolidate Basis combination generators into a single parameterized routine to reduce
  duplicated sorting/ordering logic in `src/algebra/basis.cpp`.
- Generalize Hubbard models (1D/2D/3D) to a dimension-agnostic helper using `Index`
  to reduce duplicated neighbor enumeration in `src/algebra/fermion/model/hubbard_model.h`.
- Share inner loops between serial and OpenMP matrix element routines to avoid duplication
  in `src/algebra/matrix_elements.h`.
- Merge duplicate initialization paths in `IndexedHashSet` (range vs moved vector) into a
  single helper to simplify `src/utils/indexed_hash_set.h`.
- Unify `Expression::add_to_map` overloads via forwarding and centralize repeated multiply
  and append logic in the per-algebra `expression.cpp`/`expression.h` files.
- Implement total momentum/relative position hamiltonians for Hubbard Model. 
- LinearOperator perf: avoid temporaries by adding in-place apply(out, in) and use in Sum/Difference/Composed.
- LinearOperator perf: store operands by reference/pointer or use forwarding to avoid heavy operator copies in compositions.
- LinearOperator perf: consider non-virtual/expression-template path for hot loops to reduce dispatch and allocations.
- LinearOperator perf: reduce modulo cost in Hubbard relative kinetic operator (edge/inner loop or precomputed neighbors).
- Performance: the generic path uses per-site coordinate extraction and offset vectors, which is a bit heavier than hard-coded loops. For the examples, it should be fine; if performance
  matters, you can reuse preallocated coords/offsets vectors inside the loop to avoid allocations.
- Relative position basis (direct construction): Create a new `Basis` factory method that directly
  generates relative position states as `Expression` objects. Each basis vector would store the
  Fourier superposition of momentum-space terms with phase coefficients, rather than using a
  transformation matrix approach. This would allow working directly in the relative coordinate
  representation without needing to transform Hamiltonians.
- Release-build safety (remaining): `hubbard_momentum_factorized*.h` and
  `current_operator_fixed_k.h` still assert-only check the vector size in `apply()`; switch them
  to `require_vector_dimension` from `src/numerics/linear_operator.h`.
