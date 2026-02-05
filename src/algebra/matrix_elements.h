#pragma once

#include <omp.h>

#include <utility>
#include <vector>

#include "algebra/expression.h"
#include "algebra/normal_order.h"

template <typename VectorType, typename Basis>
VectorType compute_vector_elements_serial(const Basis& basis, const Expression& A) {
  const auto& set = basis.set;
  VectorType result(set.size());
  result.zeros();
  for (size_t i = 0; i < set.size(); ++i) {
    Expression left(set[i]);
    Expression product = normal_order(left.adjoint() * A);
    {
      auto it = product.terms().find({});
      result(i) = (it != product.terms().end()) ? it->second : Expression::complex_type{};
    }
  }
  return result;
}

template <typename VectorType, typename Basis>
VectorType compute_vector_elements(const Basis& basis, const Expression& A) {
  const auto& set = basis.set;
  VectorType result(set.size());
  result.zeros();
#pragma omp parallel
  {
#pragma omp for schedule(dynamic)
    for (size_t i = 0; i < set.size(); ++i) {
      Expression left(set[i]);
      Expression product = normal_order(left.adjoint() * A);
      {
        auto it = product.terms().find({});
        result(i) = (it != product.terms().end()) ? it->second : Expression::complex_type{};
      }
    }
  }
  return result;
}

template <typename MatrixType, typename Basis>
MatrixType compute_matrix_elements_serial(const Basis& basis, const Expression& A) {
  const auto& set = basis.set;
  MatrixType result(set.size(), set.size());
  result.zeros();
  for (size_t j = 0; j < set.size(); ++j) {
    Expression right(set[j]);
    Expression product = normal_order(A * right);
    for (const auto& term : product.terms()) {
      if (set.contains(term.first)) {
        size_t i = set.index_of(term.first);
        result(i, j) = term.second;
      }
    }
  }
  return result;
}

template <typename MatrixType, typename Basis>
MatrixType compute_matrix_elements(const Basis& basis, const Expression& A) {
  const auto& set = basis.set;
  MatrixType result(set.size(), set.size());
  result.zeros();
#pragma omp parallel
  {
#pragma omp for schedule(dynamic)
    for (size_t j = 0; j < set.size(); ++j) {
      Expression right(set[j]);
      Expression product = normal_order(A * right);
      std::vector<std::pair<size_t, Expression::complex_type>> coefficients;
      coefficients.reserve(product.terms().size());
      for (const auto& term : product.terms()) {
        if (set.contains(term.first)) {
          size_t i = set.index_of(term.first);
          coefficients.emplace_back(i, term.second);
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
template <typename MatrixType, typename BasisRow, typename BasisCol>
MatrixType compute_rectangular_matrix_elements_serial(const BasisRow& row_basis,
                                                      const BasisCol& col_basis,
                                                      const Expression& A) {
  const auto& row_set = row_basis.set;
  const auto& col_set = col_basis.set;
  MatrixType result(row_set.size(), col_set.size());
  result.zeros();
  for (size_t j = 0; j < col_set.size(); ++j) {
    Expression right(col_set[j]);
    Expression product = normal_order(A * right);
    for (const auto& term : product.terms()) {
      if (row_set.contains(term.first)) {
        size_t i = row_set.index_of(term.first);
        result(i, j) = term.second;
      }
    }
  }
  return result;
}

template <typename MatrixType, typename BasisRow, typename BasisCol>
MatrixType compute_rectangular_matrix_elements(const BasisRow& row_basis, const BasisCol& col_basis,
                                               const Expression& A) {
  const auto& row_set = row_basis.set;
  const auto& col_set = col_basis.set;
  MatrixType result(row_set.size(), col_set.size());
  result.zeros();
#pragma omp parallel
  {
#pragma omp for schedule(dynamic)
    for (size_t j = 0; j < col_set.size(); ++j) {
      Expression right(col_set[j]);
      Expression product = normal_order(A * right);
      std::vector<std::pair<size_t, Expression::complex_type>> coefficients;
      coefficients.reserve(product.terms().size());
      for (const auto& term : product.terms()) {
        if (row_set.contains(term.first)) {
          size_t i = row_set.index_of(term.first);
          coefficients.emplace_back(i, term.second);
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
