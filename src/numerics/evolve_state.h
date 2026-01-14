#pragma once

#include <algorithm>
#include <armadillo>
#include <cmath>
#include <complex>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "numerics/lanczos.h"

namespace detail {

// Returns the first column of exp(-i t T_k) in the Lanczos Krylov subspace,
// i.e. y = exp(-i t T_k) e1, where T_k is real-symmetric tridiagonal.
//
// This is the standard "expmv" approach for unitary time evolution with a
// Hermitian Hamiltonian: build the Krylov basis using Lanczos on H (not -iH),
// then exponentiate the small tridiagonal in that basis.
template <typename ScalarType>
std::vector<ScalarType> expm_tridiagonal_first_column_unitary(
    const std::vector<scalar_real_t<ScalarType>>& alphas,
    const std::vector<scalar_real_t<ScalarType>>& betas, scalar_real_t<ScalarType> t) {
  using RealType = scalar_real_t<ScalarType>;

  const size_t n = alphas.size();
  std::vector<ScalarType> y(n, static_cast<ScalarType>(0));
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

  const arma::Mat<ScalarType> Q = arma::conv_to<arma::Mat<ScalarType>>::from(eigenvectors);
  const arma::Col<ScalarType> weights = Q.row(0).t();  // Q^T e1

  const ScalarType phase = -ScalarType(0, 1) * static_cast<RealType>(t);
  const arma::Col<ScalarType> evals_c = arma::conv_to<arma::Col<ScalarType>>::from(eigenvalues);
  const arma::Col<ScalarType> exp_values = arma::exp(phase * evals_c);
  const arma::Col<ScalarType> y_vec = Q * (exp_values % weights);

  for (size_t i = 0; i < n; ++i) {
    y[i] = y_vec(i);
  }
  return y;
}

}  // namespace detail

template <typename Scalar>
struct EvolutionOptions {
  size_t krylov_steps = 30;
};

struct EvolutionNoopCallback {
  template <typename... Args>
  void operator()(Args&&...) const {}
};

// Compute psi(t) = exp(-i t H) psi(0) using a Lanczos/Krylov expmv method.
//
// Requirements:
//   - H must be Hermitian with respect to the Armadillo inner product used in lanczos.h.
//   - The state vector scalar type should be complex (e.g. arma::cx_vec).
template <typename Op>
typename Op::VectorType time_evolve_state(const Op& H, const typename Op::VectorType& psi0,
                                          scalar_real_t<typename Op::ScalarType> t,
                                          EvolutionOptions<typename Op::ScalarType> opts = {}) {
  using ScalarType = typename Op::ScalarType;
  using RealType = scalar_real_t<ScalarType>;
  static_assert(std::is_same_v<ScalarType, std::complex<RealType>>,
                "time_evolve_state expects a complex scalar type (e.g. arma::cx_vec)");

  if (psi0.n_elem == 0) {
    return psi0;
  }

  const RealType psi_norm = arma::norm(psi0);
  if (psi_norm <= breakdown_tolerance<ScalarType>()) {
    return psi0;
  }

  size_t k = opts.krylov_steps;
  const size_t dim = H.dimension();
  if (k == 0 || k > dim) {
    k = dim;
  }
  if (k == 0) {
    return typename Op::VectorType(psi0.n_elem, arma::fill::zeros);
  }

  return solve(H, psi0, k, [t](const std::vector<RealType>& alphas,
                              const std::vector<RealType>& betas) {
    return detail::expm_tridiagonal_first_column_unitary<ScalarType>(alphas, betas, t);
  });
}

// Convenience wrapper for stroboscopic evolution with a time-independent Hamiltonian:
// psi(t1) obtained by applying exp(-i dt H) repeatedly.
//
// This is useful when you want intermediate states at uniform spacing without repeatedly
// recomputing the Krylov decomposition for a large t.
template <typename Op, typename Callback = EvolutionNoopCallback>
typename Op::VectorType time_evolve_state_steps(
    const Op& H, typename Op::VectorType psi, scalar_real_t<typename Op::ScalarType> t0,
    scalar_real_t<typename Op::ScalarType> t1, scalar_real_t<typename Op::ScalarType> dt,
    EvolutionOptions<typename Op::ScalarType> opts = {}, Callback callback = {}) {
  using ScalarType = typename Op::ScalarType;
  using RealType = scalar_real_t<ScalarType>;

  if (dt == static_cast<RealType>(0) || (t1 - t0) == static_cast<RealType>(0)) {
    callback(t0, psi);
    return psi;
  }

  const RealType direction = (t1 >= t0) ? static_cast<RealType>(1) : static_cast<RealType>(-1);
  dt = std::abs(dt) * direction;

  RealType t = t0;
  callback(t, psi);
  while ((direction > 0 && t < t1) || (direction < 0 && t > t1)) {
    const RealType step = (direction > 0) ? std::min(dt, t1 - t) : std::max(dt, t1 - t);
    psi = time_evolve_state(H, psi, step, opts);
    t += step;
    callback(t, psi);
  }
  return psi;
}
