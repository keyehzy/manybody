#pragma once

#include <armadillo>
#include <cmath>
#include <numbers>
#include <stdexcept>
#include <vector>

namespace kpm {

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

/// Compute Chebyshev moments for the complementary projector Q = 1 - P
/// These have opposite sign for m > 0
inline std::vector<double> complementary_projector_moments(size_t M, double epsilon) {
  std::vector<double> mu(M + 1);
  const double acos_eps = std::acos(epsilon);

  mu[0] = acos_eps / std::numbers::pi;
  for (size_t m = 1; m <= M; ++m) {
    const double md = static_cast<double>(m);
    mu[m] = 2.0 * std::sin(md * acos_eps) / (md * std::numbers::pi);
  }
  return mu;
}

/// Rescale Hamiltonian to have spectrum in [-1, 1]
/// H_scaled = (H - b) / a, where a = (E_max - E_min) / 2, b = (E_max + E_min) / 2
struct Rescaling {
  double a;  // Scale factor
  double b;  // Shift

  /// Rescale an energy value
  double rescale(double E) const { return (E - b) / a; }

  /// Inverse rescale
  double inverse(double E_scaled) const { return E_scaled * a + b; }
};

/// Estimate spectral bounds using power iteration
inline Rescaling estimate_rescaling(const arma::mat& H, double padding = 0.01) {
  arma::vec eigvals = arma::eig_sym(H);
  const double E_min = eigvals.min();
  const double E_max = eigvals.max();

  const double a = (E_max - E_min) / 2.0 * (1.0 + padding);
  const double b = (E_max + E_min) / 2.0;

  return Rescaling{a, b};
}

/// Rescale matrix
inline arma::mat rescale_hamiltonian(const arma::mat& H, const Rescaling& rescaling) {
  return (H - rescaling.b * arma::eye(H.n_rows, H.n_cols)) / rescaling.a;
}

/// KPM expansion of projector onto states below E_fermi
/// P_KPM = Σ_m g_m μ_m T_m(H_scaled)
///
/// Uses Chebyshev recurrence: T_0 = I, T_1 = H, T_{m+1} = 2H T_m - T_{m-1}
class KPMProjector {
 public:
  KPMProjector(const arma::mat& H, size_t expansion_order, double E_fermi = 0.0,
               double padding = 0.01)
      : M_(expansion_order), rescaling_(estimate_rescaling(H, padding)) {
    // Rescale Hamiltonian
    H_scaled_ = rescale_hamiltonian(H, rescaling_);

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
  arma::vec apply(const arma::vec& v) const {
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

 private:
  size_t M_;
  Rescaling rescaling_;
  arma::mat H_scaled_;
  std::vector<double> moments_;
  std::vector<double> kernel_;
  std::vector<double> damped_moments_;
};

/// KPM expansion with sparse matrix support (for large systems)
class SparseKPMProjector {
 public:
  SparseKPMProjector(const arma::sp_mat& H, size_t expansion_order, double E_min, double E_max,
                     double E_fermi = 0.0, double padding = 0.01)
      : M_(expansion_order) {
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

  arma::vec apply(const arma::vec& v) const {
    arma::vec T_prev = v;
    arma::vec result = damped_moments_[0] * T_prev;

    if (M_ == 0) return result;

    arma::vec T_curr = H_scaled_ * v;
    result += damped_moments_[1] * T_curr;

    for (size_t m = 2; m <= M_; ++m) {
      arma::vec T_next = 2.0 * (H_scaled_ * T_curr) - T_prev;
      result += damped_moments_[m] * T_next;
      T_prev = T_curr;
      T_curr = T_next;
    }

    return result;
  }

 private:
  size_t M_;
  Rescaling rescaling_;
  arma::sp_mat H_scaled_;
  std::vector<double> moments_;
  std::vector<double> kernel_;
  std::vector<double> damped_moments_;
};

}  // namespace kpm
