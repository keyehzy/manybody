#pragma once

#include <armadillo>
#include <cmath>
#include <numbers>
#include <stdexcept>
#include <vector>

#include "numerics/linear_operator.h"
#include "numerics/rescaling.h"

namespace kpm {

// Re-export rescaling types for backward compatibility
using Rescaling = rescaling::Rescaling;

/// Jackson kernel damping factors to reduce Gibbs oscillations
/// g_n = ((N - n + 1) cos(πn/(N+1)) + sin(πn/(N+1)) cot(π/(N+1))) / (N + 1)
inline std::vector<double> jackson_kernel(size_t N) {
  std::vector<double> g(N + 1);
  const double denom = static_cast<double>(N + 1);
  const double cot_val = 1.0 / std::tan(std::numbers::pi / denom);

  for (size_t n = 0; n <= N; ++n) {
    const double nd = static_cast<double>(n);
    const double arg = std::numbers::pi * nd / denom;
    g[n] = ((denom - nd) * std::cos(arg) + std::sin(arg) * cot_val) / denom;
  }
  return g;
}

/// Lorentz kernel damping factors (alternative to Jackson)
/// g_n = sinh(λ(1 - n/N)) / sinh(λ)
inline std::vector<double> lorentz_kernel(size_t N, double lambda = 4.0) {
  std::vector<double> g(N + 1);
  const double sinh_lambda = std::sinh(lambda);

  for (size_t n = 0; n <= N; ++n) {
    const double nd = static_cast<double>(n);
    const double Nd = static_cast<double>(N);
    g[n] = std::sinh(lambda * (1.0 - nd / Nd)) / sinh_lambda;
  }
  return g;
}

/// Compute Chebyshev moments of the step function θ(ε - x)
/// μ_m(ε) for the projector P = θ(E_F - H)
///
/// μ_0 = 1 - arccos(ε)/π
/// μ_m = -2 sin(m arccos(ε)) / (mπ)  for m > 0
inline std::vector<double> projector_moments(size_t M, double epsilon) {
  std::vector<double> mu(M + 1);
  const double acos_eps = std::acos(epsilon);

  mu[0] = 1.0 - acos_eps / std::numbers::pi;
  for (size_t m = 1; m <= M; ++m) {
    const double md = static_cast<double>(m);
    mu[m] = -2.0 * std::sin(md * acos_eps) / (md * std::numbers::pi);
  }
  return mu;
}

/// KPM expansion of projector onto states below E_fermi
/// P_KPM = Σ_m g_m μ_m T_m(H_scaled)
///
/// Uses Chebyshev recurrence: T_0 = I, T_1 = H, T_{m+1} = 2H T_m - T_{m-1}
class KPMProjector final : public LinearOperator<arma::vec> {
 public:
  using VectorType = arma::vec;
  using ScalarType = double;

  KPMProjector(const arma::mat& H, size_t expansion_order, double E_fermi = 0.0,
               double padding = 0.01)
      : M_(expansion_order),
        dim_(static_cast<size_t>(H.n_rows)),
        rescaling_(rescaling::estimate_rescaling(H, padding)) {
    // Rescale Hamiltonian
    H_scaled_ = rescaling::rescale_hamiltonian(H, rescaling_);

    // Rescale Fermi energy
    const double epsilon = rescaling_.rescale(E_fermi);
    if (epsilon < -1.0 || epsilon > 1.0) {
      throw std::runtime_error("Fermi energy outside rescaled spectrum");
    }

    // Compute moments and kernel
    moments_ = projector_moments(M_, epsilon);
    kernel_ = jackson_kernel(M_);

    // Combine into damped moments
    damped_moments_.resize(M_ + 1);
    for (size_t m = 0; m <= M_; ++m) {
      damped_moments_[m] = kernel_[m] * moments_[m];
    }
  }

  /// Apply projector to a vector using Chebyshev recurrence
  VectorType apply(const VectorType& v) const override {
    // T_0(H)|v⟩ = |v⟩
    arma::vec T_prev = v;
    arma::vec result = damped_moments_[0] * T_prev;

    if (M_ == 0) return result;

    // T_1(H)|v⟩ = H|v⟩
    arma::vec T_curr = H_scaled_ * v;
    result += damped_moments_[1] * T_curr;

    // Recurrence: T_{m+1} = 2H T_m - T_{m-1}
    for (size_t m = 2; m <= M_; ++m) {
      arma::vec T_next = 2.0 * (H_scaled_ * T_curr) - T_prev;
      result += damped_moments_[m] * T_next;

      T_prev = T_curr;
      T_curr = T_next;
    }

    return result;
  }

  /// Apply projector to a complex vector using Chebyshev recurrence
  arma::cx_vec apply(const arma::cx_vec& v) const {
    arma::cx_vec T_prev = v;
    arma::cx_vec result = damped_moments_[0] * T_prev;

    if (M_ == 0) return result;

    arma::cx_vec T_curr = H_scaled_ * v;
    result += damped_moments_[1] * T_curr;

    for (size_t m = 2; m <= M_; ++m) {
      arma::cx_vec T_next = 2.0 * (H_scaled_ * T_curr) - T_prev;
      result += damped_moments_[m] * T_next;

      T_prev = T_curr;
      T_curr = T_next;
    }

    return result;
  }

  /// Build the full projector matrix (expensive, for verification)
  arma::mat build_matrix() const {
    const size_t n = H_scaled_.n_rows;
    arma::mat P(n, n, arma::fill::zeros);

    // T_0 = I
    arma::mat T_prev = arma::eye(n, n);
    P += damped_moments_[0] * T_prev;

    if (M_ == 0) return P;

    // T_1 = H
    arma::mat T_curr = H_scaled_;
    P += damped_moments_[1] * T_curr;

    // Recurrence
    for (size_t m = 2; m <= M_; ++m) {
      arma::mat T_next = 2.0 * H_scaled_ * T_curr - T_prev;
      P += damped_moments_[m] * T_next;

      T_prev = T_curr;
      T_curr = T_next;
    }

    return P;
  }

  size_t expansion_order() const { return M_; }
  const Rescaling& rescaling() const { return rescaling_; }
  size_t dimension() const override { return dim_; }

 private:
  size_t M_;
  size_t dim_;
  Rescaling rescaling_;
  arma::mat H_scaled_;
  std::vector<double> moments_;
  std::vector<double> kernel_;
  std::vector<double> damped_moments_;
};

/// KPM expansion with sparse matrix support (for large systems)
class SparseKPMProjector final : public LinearOperator<arma::vec> {
 public:
  using VectorType = arma::vec;
  using ScalarType = double;

  SparseKPMProjector(const arma::sp_mat& H, size_t expansion_order, double E_min, double E_max,
                     double E_fermi = 0.0, double padding = 0.01)
      : M_(expansion_order), dim_(static_cast<size_t>(H.n_rows)) {
    // Compute rescaling
    const double a = (E_max - E_min) / 2.0 * (1.0 + padding);
    const double b = (E_max + E_min) / 2.0;
    rescaling_ = Rescaling{a, b};

    // Rescale Hamiltonian (sparse)
    H_scaled_ = (H - b * arma::speye<arma::sp_mat>(H.n_rows, H.n_cols)) / a;

    // Rescale Fermi energy
    const double epsilon = rescaling_.rescale(E_fermi);

    // Compute damped moments
    moments_ = projector_moments(M_, epsilon);
    kernel_ = jackson_kernel(M_);
    damped_moments_.resize(M_ + 1);
    for (size_t m = 0; m <= M_; ++m) {
      damped_moments_[m] = kernel_[m] * moments_[m];
    }
  }

  VectorType apply(const VectorType& v) const override {
    VectorType T_prev = v;
    VectorType result = damped_moments_[0] * T_prev;

    if (M_ == 0) return result;

    VectorType T_curr = H_scaled_ * v;
    result += damped_moments_[1] * T_curr;

    for (size_t m = 2; m <= M_; ++m) {
      VectorType T_next = 2.0 * (H_scaled_ * T_curr) - T_prev;
      result += damped_moments_[m] * T_next;
      T_prev = T_curr;
      T_curr = T_next;
    }

    return result;
  }

  size_t dimension() const override { return dim_; }

 private:
  size_t M_;
  size_t dim_;
  Rescaling rescaling_;
  arma::sp_mat H_scaled_;
  std::vector<double> moments_;
  std::vector<double> kernel_;
  std::vector<double> damped_moments_;
};

}  // namespace kpm
