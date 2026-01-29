#pragma once

#include <armadillo>
#include <cmath>
#include <complex>
#include <numbers>
#include <vector>

#include "numerics/kpm.h"
#include "numerics/lanczos.h"
#include "numerics/linear_operator.h"

namespace topological {

namespace detail {
inline arma::mat default_position_operator(size_t n_sites) {
  arma::mat X(n_sites, n_sites, arma::fill::zeros);
  for (size_t i = 0; i < n_sites; ++i) {
    X(i, i) = static_cast<double>(i);
  }
  return X;
}

inline arma::cx_mat to_complex(const arma::mat& real_matrix) {
  return arma::conv_to<arma::cx_mat>::from(real_matrix);
}
}  // namespace detail

/// 1D Topological Marker for SSH-like models (Class AIII)
///
/// From Paper 2, the 1D topological operator is:
///   Ĉ_1D = N_1 W [Q x̂ P + P x̂ Q]
///
/// where:
///   - P = projector onto occupied states (E < E_F)
///   - Q = I - P = projector onto empty states
///   - x̂ = position operator
///   - W = σ_z = chiral operator (+1 on A sublattice, -1 on B sublattice)
///   - N_1 = 2πi (normalization for 1D)
///
/// The local marker C(r) = ⟨r|Ĉ|r⟩ gives the local contribution to the winding number.
/// The spatially averaged marker Σ_r C(r) / L gives the winding number.

/// Compute the 1D topological marker using exact diagonalization
/// Following Paper 2, Eq. 2 for D=1
struct Marker1D {
  arma::mat P;     // Projector onto occupied states
  arma::mat Q;     // Complementary projector
  arma::mat W;     // Chiral operator (σ_z)
  arma::cx_mat X;  // Position operator (allow complex for PBC exp operator)
  size_t n_sites;  // Total number of sites

  Marker1D(const arma::mat& H, const arma::mat& chiral, double E_fermi = 0.0)
      : Marker1D(H, chiral, detail::default_position_operator(H.n_rows), E_fermi) {}

  Marker1D(const arma::mat& H, const arma::mat& chiral, const arma::mat& position,
           double E_fermi = 0.0)
      : Marker1D(H, chiral, detail::to_complex(position), E_fermi) {}

  Marker1D(const arma::mat& H, const arma::mat& chiral, const arma::cx_mat& position,
           double E_fermi = 0.0) {
    n_sites = H.n_rows;
    W = chiral;
    X = position;

    // Diagonalize H
    arma::vec eigvals;
    arma::mat eigvecs;
    arma::eig_sym(eigvals, eigvecs, H);

    // Build projector P onto occupied states (E < E_fermi)
    P = arma::mat(n_sites, n_sites, arma::fill::zeros);
    for (size_t i = 0; i < n_sites; ++i) {
      if (eigvals(i) < E_fermi) {
        P += eigvecs.col(i) * eigvecs.col(i).t();
      }
    }
    Q = arma::eye(n_sites, n_sites) - P;
  }

  /// Compute the marker operator matrix
  /// Ĉ = N_1 W (Q X P + P X Q)
  /// N_1 = 2πi for 1D
  arma::cx_mat marker_operator() const {
    const std::complex<double> N1(0.0, 2.0 * std::numbers::pi);

    // Q X P + P X Q
    arma::cx_mat Qc = arma::conv_to<arma::cx_mat>::from(Q);
    arma::cx_mat Pc = arma::conv_to<arma::cx_mat>::from(P);
    arma::cx_mat Wc = arma::conv_to<arma::cx_mat>::from(W);
    arma::cx_mat sum = Qc * X * Pc + Pc * X * Qc;

    // W * (QXP + PXQ) and multiply by N_1
    return N1 * (Wc * sum);
  }

  /// Local marker: C(r) = ⟨r|Ĉ|r⟩
  std::vector<double> local_marker() const {
    arma::cx_mat C = marker_operator();
    std::vector<double> marker(n_sites);
    for (size_t r = 0; r < n_sites; ++r) {
      marker[r] = std::imag(C(r, r));
    }
    return marker;
  }

  /// Spatially averaged marker = (1/L) Σ_r C(r) = (1/L) Tr(Ĉ)
  /// This should give the winding number
  double average_marker() const {
    arma::cx_mat C = marker_operator();
    return std::imag(arma::trace(C)) / static_cast<double>(n_sites);
  }

  /// Total marker = Tr(Ĉ) (not normalized)
  double total_marker() const {
    arma::cx_mat C = marker_operator();
    return std::imag(arma::trace(C));
  }
};

/// KPM-based 1D topological marker
/// Uses KPM to approximate the projector P
struct Marker1D_KPM {
  kpm::KPMProjector P_kpm;
  arma::mat W;
  arma::cx_mat X;
  size_t n_sites;

  Marker1D_KPM(const arma::mat& H, const arma::mat& chiral, size_t kpm_order, double E_fermi = 0.0)
      : Marker1D_KPM(H, chiral, detail::default_position_operator(H.n_rows), kpm_order, E_fermi) {}

  Marker1D_KPM(const arma::mat& H, const arma::mat& chiral, const arma::mat& position,
               size_t kpm_order, double E_fermi = 0.0)
      : Marker1D_KPM(H, chiral, detail::to_complex(position), kpm_order, E_fermi) {}

  Marker1D_KPM(const arma::mat& H, const arma::mat& chiral, const arma::cx_mat& position,
               size_t kpm_order, double E_fermi = 0.0)
      : P_kpm(H, kpm_order, E_fermi), W(chiral), X(position), n_sites(H.n_rows) {}

  arma::cx_mat marker_operator() const {
    const std::complex<double> N1(0.0, 2.0 * std::numbers::pi);

    arma::mat P = P_kpm.build_matrix();
    arma::cx_mat Pc = arma::conv_to<arma::cx_mat>::from(P);
    arma::cx_mat Qc = arma::eye<arma::cx_mat>(n_sites, n_sites) - Pc;
    arma::cx_mat Wc = arma::conv_to<arma::cx_mat>::from(W);

    arma::cx_mat sum = Qc * X * Pc + Pc * X * Qc;
    return N1 * (Wc * sum);
  }

  std::vector<double> local_marker() const {
    arma::cx_mat C = marker_operator();
    std::vector<double> marker(n_sites);
    for (size_t r = 0; r < n_sites; ++r) {
      marker[r] = std::imag(C(r, r));
    }
    return marker;
  }

  double average_marker() const {
    arma::cx_mat C = marker_operator();
    return std::imag(arma::trace(C)) / static_cast<double>(n_sites);
  }

  double total_marker() const {
    arma::cx_mat C = marker_operator();
    return std::imag(arma::trace(C));
  }

  size_t expansion_order() const { return P_kpm.expansion_order(); }
};

/// Lanczos-based 1D topological marker
/// Uses Lanczos-Ritz to approximate the projector P
struct Marker1D_Lanczos {
  const arma::mat& H;
  arma::mat W;
  arma::cx_mat X;
  size_t n_sites;
  double E_fermi;

  Marker1D_Lanczos(const arma::mat& H_in, const arma::mat& chiral, double E_fermi_in = 0.0)
      : Marker1D_Lanczos(H_in, chiral, detail::default_position_operator(H_in.n_rows), E_fermi_in) {
  }

  Marker1D_Lanczos(const arma::mat& H_in, const arma::mat& chiral, const arma::mat& position,
                   double E_fermi_in = 0.0)
      : Marker1D_Lanczos(H_in, chiral, detail::to_complex(position), E_fermi_in) {}

  Marker1D_Lanczos(const arma::mat& H_in, const arma::mat& chiral, const arma::cx_mat& position,
                   double E_fermi_in = 0.0)
      : H(H_in), W(chiral), X(position), n_sites(H_in.n_rows), E_fermi(E_fermi_in) {}

  /// Build projector using Lanczos from a random starting vector
  std::pair<arma::mat, size_t> build_projector(size_t max_krylov) const {
    arma::vec start = arma::randn<arma::vec>(n_sites);
    start /= arma::norm(start);

    MatrixOperator op(H);
    auto decomp = lanczos_pass_one(op, start, max_krylov);

    const size_t m = decomp.steps_taken;
    if (m == 0) {
      return {arma::mat(n_sites, n_sites, arma::fill::zeros), 0};
    }

    // Build tridiagonal matrix T
    arma::mat T(m, m, arma::fill::zeros);
    for (size_t i = 0; i < m; ++i) {
      T(i, i) = decomp.alphas[i];
      if (i + 1 < m) {
        T(i, i + 1) = decomp.betas[i];
        T(i + 1, i) = decomp.betas[i];
      }
    }

    // Diagonalize T to get Ritz values and vectors
    arma::vec ritz_vals;
    arma::mat ritz_vecs_T;
    arma::eig_sym(ritz_vals, ritz_vecs_T, T);

    // Reconstruct Ritz vectors in full space and build projector
    arma::mat P(n_sites, n_sites, arma::fill::zeros);
    for (size_t k = 0; k < m; ++k) {
      std::vector<double> y_k(m);
      for (size_t i = 0; i < m; ++i) {
        y_k[i] = ritz_vecs_T(i, k) * decomp.b_norm;
      }
      arma::vec rv = lanczos_pass_two(op, start, decomp, y_k);
      double norm = arma::norm(rv);
      if (norm > 1e-10) {
        rv /= norm;
        if (ritz_vals(k) < E_fermi) {
          P += rv * rv.t();
        }
      }
    }

    return {P, m};
  }

  /// Compute marker with given Krylov dimension
  std::pair<arma::cx_mat, size_t> marker_operator(size_t max_krylov) const {
    auto [P, m] = build_projector(max_krylov);

    const std::complex<double> N1(0.0, 2.0 * std::numbers::pi);
    arma::cx_mat Pc = arma::conv_to<arma::cx_mat>::from(P);
    arma::cx_mat Qc = arma::eye<arma::cx_mat>(n_sites, n_sites) - Pc;
    arma::cx_mat Wc = arma::conv_to<arma::cx_mat>::from(W);
    arma::cx_mat sum = Qc * X * Pc + Pc * X * Qc;

    return {N1 * (Wc * sum), m};
  }

  std::pair<std::vector<double>, size_t> local_marker(size_t max_krylov) const {
    auto [C, m] = marker_operator(max_krylov);
    std::vector<double> marker(n_sites);
    for (size_t r = 0; r < n_sites; ++r) {
      marker[r] = std::imag(C(r, r));
    }
    return {marker, m};
  }

  std::pair<double, size_t> average_marker(size_t max_krylov) const {
    auto [C, m] = marker_operator(max_krylov);
    return {std::imag(arma::trace(C)) / static_cast<double>(n_sites), m};
  }

  std::pair<double, size_t> total_marker(size_t max_krylov) const {
    auto [C, m] = marker_operator(max_krylov);
    return {std::imag(arma::trace(C)), m};
  }

  /// Compute with convergence tracking
  struct ConvergenceResult {
    double marker;
    size_t krylov_dim;
    double error_estimate;
    bool converged;
  };

  ConvergenceResult compute_with_convergence(double tolerance = 1e-4,
                                             size_t max_krylov = 500) const {
    double prev_marker = 0.0;

    for (size_t m = 10; m <= max_krylov; m += 10) {
      auto [marker, actual_m] = average_marker(m);
      double error = std::abs(marker - prev_marker);

      if (error < tolerance && m > 10) {
        return {marker, actual_m, error, true};
      }

      prev_marker = marker;
    }

    auto [marker, actual_m] = average_marker(max_krylov);
    return {marker, actual_m, std::abs(marker - prev_marker), false};
  }
};

/// IPR analysis for Ritz vectors (from Paper 1)
struct IPRAnalysis {
  static double ipr_krylov(const arma::vec& initial_state,
                           const std::vector<arma::vec>& ritz_vectors) {
    double ipr = 0.0;
    double norm_sq = 0.0;

    for (const auto& rv : ritz_vectors) {
      double overlap_sq = std::pow(arma::dot(rv, initial_state), 2);
      ipr += overlap_sq * overlap_sq;
      norm_sq += overlap_sq;
    }

    return (norm_sq > 1e-10) ? ipr / (norm_sq * norm_sq) : 0.0;
  }

  static double ipr_local(const arma::vec& ritz_vector) {
    double ipr = 0.0;
    for (size_t i = 0; i < ritz_vector.n_elem; ++i) {
      double val = ritz_vector(i);
      ipr += val * val * val * val;
    }
    return ipr;
  }

  static double ipr_local_average(const std::vector<arma::vec>& ritz_vectors) {
    if (ritz_vectors.empty()) return 0.0;
    double sum = 0.0;
    for (const auto& rv : ritz_vectors) {
      sum += ipr_local(rv);
    }
    return sum / static_cast<double>(ritz_vectors.size());
  }
};

}  // namespace topological
