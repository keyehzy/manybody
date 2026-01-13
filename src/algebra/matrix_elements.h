#pragma once

#include <omp.h>

#include <utility>
#include <vector>

#include "algebra/expression.h"
#include "algebra/normal_order.h"

template <typename VectorType, typename Basis>
VectorType compute_vector_elements_serial(const Basis& basis, const Expression& A,
                                          NormalOrderer& orderer) {
  const auto& set = basis.set;
  VectorType result(set.size());
  result.zeros();
  for (size_t i = 0; i < set.size(); ++i) {
    Expression left(set[i]);
    Expression product = orderer.normal_order(left.adjoint() * A);
    result(i) = product.hashmap[{}];
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
    NormalOrderer orderer;
#pragma omp for schedule(dynamic)
    for (size_t i = 0; i < set.size(); ++i) {
      Expression left(set[i]);
      Expression product = orderer.normal_order(left.adjoint() * A);
      result(i) = product.hashmap[{}];
    }
  }
  return result;
}

template <typename MatrixType, typename Basis>
MatrixType compute_matrix_elements_serial(const Basis& basis, const Expression& A,
                                          NormalOrderer& orderer) {
  const auto& set = basis.set;
  MatrixType result(set.size(), set.size());
  result.zeros();
  for (size_t j = 0; j < set.size(); ++j) {
    Expression right(set[j]);
    Expression product = orderer.normal_order(A * right);
    for (const auto& term : product.hashmap) {
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
    NormalOrderer orderer;
#pragma omp for schedule(dynamic)
    for (size_t j = 0; j < set.size(); ++j) {
      Expression right(set[j]);
      Expression product = orderer.normal_order(A * right);
      std::vector<std::pair<size_t, Expression::complex_type>> coefficients;
      coefficients.reserve(product.hashmap.size());
      for (const auto& term : product.hashmap) {
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
