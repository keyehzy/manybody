#pragma once

#include <algorithm>
#include <armadillo>
#include <stdexcept>
#include <vector>

#include "lanczos.h"

namespace exp_detail {
template <typename RealType>
std::vector<RealType> exp_tridiagonal_first_column(const std::vector<RealType>& alphas,
                                                   const std::vector<RealType>& betas) {
  const size_t n = alphas.size();
  std::vector<RealType> y(n, static_cast<RealType>(0));
  if (n == 0) {
    return y;
  }

  arma::Mat<RealType> T(n, n, arma::fill::zeros);
  for (size_t i = 0; i < n; ++i) {
    T(i, i) = alphas[i];
  }
  for (size_t i = 0; i < betas.size(); ++i) {
    T(i, i + 1) = betas[i];
    T(i + 1, i) = betas[i];
  }

  arma::Col<RealType> eigenvalues;
  arma::Mat<RealType> eigenvectors;
  if (!arma::eig_sym(eigenvalues, eigenvectors, T)) {
    throw std::runtime_error("Eigenvalue decomposition of T_k failed");
  }

  const arma::Col<RealType> weights = eigenvectors.row(0).t();
  const arma::Col<RealType> exp_values = arma::exp(eigenvalues);
  const arma::Col<RealType> y_vec = eigenvectors * (exp_values % weights);
  for (size_t i = 0; i < n; ++i) {
    y[i] = y_vec(i);
  }
  return y;
}
}  // namespace exp_detail

template <typename Scalar>
struct ExpOptions {
  size_t krylov_steps = 30;
};

template <typename Op>
struct Exp final : LinearOperator<typename Op::VectorType> {
  using VectorType = typename Op::VectorType;
  using ScalarType = typename Op::ScalarType;
  using RealType = scalar_real_t<ScalarType>;
  using Options = ExpOptions<ScalarType>;

  Exp(Op op, Options options = {}) : op_(std::move(op)), options_(options) {}

  VectorType apply(const VectorType& v) const override {
    if (v.n_elem == 0) {
      return v;
    }

    const RealType v_norm = arma::norm(v);
    if (v_norm <= breakdown_tolerance<ScalarType>()) {
      return v;
    }

    size_t k = options_.krylov_steps;
    const size_t dimension = op_.dimension();
    if (k == 0 || k > dimension) {
      k = dimension;
    }
    if (k == 0) {
      return VectorType(v.n_elem, arma::fill::zeros);
    }

    return solve(op_, v, k,
                 [](const std::vector<RealType>& alphas, const std::vector<RealType>& betas) {
                   return exp_detail::exp_tridiagonal_first_column<RealType>(alphas, betas);
                 });
  }

  size_t dimension() const override { return op_.dimension(); }

 private:
  Op op_;
  Options options_;
};
