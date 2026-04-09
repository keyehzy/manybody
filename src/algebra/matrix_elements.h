#pragma once

#include <omp.h>

#include <utility>
#include <vector>

template <typename VectorType, typename Basis, typename Expression>
VectorType compute_vector_elements_serial(const Basis& basis, const Expression& A) {
  using complex_type = typename Expression::complex_type;
  const auto& set = basis.set;
  VectorType result(set.size());
  result.zeros();
  for (size_t i = 0; i < set.size(); ++i) {
    Expression left(set[i]);
    Expression product = canonicalize(adjoint(left) * A);
    {
      auto it = product.terms().find({});
      const complex_type normalization = basis.state_normalization(i);
      result(i) = (it != product.terms().end()) ? it->second * normalization : complex_type{};
    }
  }
  return result;
}

template <typename VectorType, typename Basis, typename Expression>
VectorType compute_vector_elements(const Basis& basis, const Expression& A) {
  using complex_type = typename Expression::complex_type;
  const auto& set = basis.set;
  VectorType result(set.size());
  result.zeros();
#pragma omp parallel
  {
#pragma omp for schedule(dynamic)
    for (size_t i = 0; i < set.size(); ++i) {
      Expression left(set[i]);
      Expression product = canonicalize(adjoint(left) * A);
      {
        auto it = product.terms().find({});
        const complex_type normalization = basis.state_normalization(i);
        result(i) = (it != product.terms().end()) ? it->second * normalization : complex_type{};
      }
    }
  }
  return result;
}

template <typename MatrixType, typename Basis, typename Expression>
MatrixType compute_matrix_elements_serial(const Basis& basis, const Expression& A) {
  using complex_type = typename Expression::complex_type;
  const auto& set = basis.set;
  MatrixType result(set.size(), set.size());
  result.zeros();
  for (size_t j = 0; j < set.size(); ++j) {
    Expression right(set[j]);
    Expression product = canonicalize(A * right);
    const complex_type right_normalization = basis.state_normalization(j);
    for (const auto& term : product.terms()) {
      if (set.contains(term.first)) {
        size_t i = set.index_of(term.first);
        const complex_type left_normalization = basis.state_normalization(i);
        result(i, j) = term.second * right_normalization / left_normalization;
      }
    }
  }
  return result;
}

template <typename MatrixType, typename Basis, typename Expression>
MatrixType compute_matrix_elements(const Basis& basis, const Expression& A) {
  using complex_type = typename Expression::complex_type;
  const auto& set = basis.set;
  MatrixType result(set.size(), set.size());
  result.zeros();
#pragma omp parallel
  {
#pragma omp for schedule(dynamic)
    for (size_t j = 0; j < set.size(); ++j) {
      Expression right(set[j]);
      Expression product = canonicalize(A * right);
      const complex_type right_normalization = basis.state_normalization(j);
      std::vector<std::pair<size_t, complex_type>> coefficients;
      coefficients.reserve(product.terms().size());
      for (const auto& term : product.terms()) {
        if (set.contains(term.first)) {
          size_t i = set.index_of(term.first);
          const complex_type left_normalization = basis.state_normalization(i);
          coefficients.emplace_back(i, term.second * right_normalization / left_normalization);
        }
      }
#pragma omp critical
      {
        for (const auto& [i, val] : coefficients) {
          result(i, j) = val;
        }
      }
    }
  }
  return result;
}

// Compute matrix elements of an operator that maps from one basis to another.
// This is used for operators like the current operator J(Q) which maps
// from momentum sector K to K+Q.
template <typename MatrixType, typename BasisRow, typename BasisCol, typename Expression>
MatrixType compute_rectangular_matrix_elements_serial(const BasisRow& row_basis,
                                                      const BasisCol& col_basis,
                                                      const Expression& A) {
  using complex_type = typename Expression::complex_type;
  const auto& row_set = row_basis.set;
  const auto& col_set = col_basis.set;
  MatrixType result(row_set.size(), col_set.size());
  result.zeros();
  for (size_t j = 0; j < col_set.size(); ++j) {
    Expression right(col_set[j]);
    Expression product = canonicalize(A * right);
    const complex_type right_normalization = col_basis.state_normalization(j);
    for (const auto& term : product.terms()) {
      if (row_set.contains(term.first)) {
        size_t i = row_set.index_of(term.first);
        const complex_type left_normalization = row_basis.state_normalization(i);
        result(i, j) = term.second * right_normalization / left_normalization;
      }
    }
  }
  return result;
}

template <typename MatrixType, typename BasisRow, typename BasisCol, typename Expression>
MatrixType compute_rectangular_matrix_elements(const BasisRow& row_basis, const BasisCol& col_basis,
                                               const Expression& A) {
  using complex_type = typename Expression::complex_type;
  const auto& row_set = row_basis.set;
  const auto& col_set = col_basis.set;
  MatrixType result(row_set.size(), col_set.size());
  result.zeros();
#pragma omp parallel
  {
#pragma omp for schedule(dynamic)
    for (size_t j = 0; j < col_set.size(); ++j) {
      Expression right(col_set[j]);
      Expression product = canonicalize(A * right);
      const complex_type right_normalization = col_basis.state_normalization(j);
      std::vector<std::pair<size_t, complex_type>> coefficients;
      coefficients.reserve(product.terms().size());
      for (const auto& term : product.terms()) {
        if (row_set.contains(term.first)) {
          size_t i = row_set.index_of(term.first);
          const complex_type left_normalization = row_basis.state_normalization(i);
          coefficients.emplace_back(i, term.second * right_normalization / left_normalization);
        }
      }
#pragma omp critical
      {
        for (const auto& [i, val] : coefficients) {
          result(i, j) = val;
        }
      }
    }
  }
  return result;
}
