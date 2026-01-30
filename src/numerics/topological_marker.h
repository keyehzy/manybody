#pragma once

#include <armadillo>
#include <cmath>
#include <complex>
#include <numbers>
#include <random>
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

  std::vector<double> local_marker_cells() const {
    arma::cx_mat C = marker_operator();
    const size_t num_cells = n_sites / 2;
    std::vector<double> marker(num_cells);
    const Index idx({2, num_cells});
    for (size_t n = 0; n < num_cells; ++n) {
      const size_t A = idx({0, n});
      const size_t B = idx({1, n});
      marker[n] = std::imag(C(A, A) + C(B, B)) / 2.0;
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
/// Uses KPM to approximate the projector P via vector applications
///
/// Instead of building the full projector matrix, we use the identity:
///   C|r⟩ = N1 W [X u + P X|r⟩ - 2 P X u], where u = P|r⟩
///
/// This allows computing local markers using only P.apply() calls.
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

  /// Compute C(r,r) = ⟨r|Ĉ|r⟩ for a single site using vector formulation
  /// C|r⟩ = N1 W [X u + P X|r⟩ - 2 P X u], where u = P|r⟩
  double local_marker_at(size_t r) const {
    const std::complex<double> N1(0.0, 2.0 * std::numbers::pi);

    // Basis vector |r⟩
    arma::vec e_r(n_sites, arma::fill::zeros);
    e_r(r) = 1.0;

    // u = P|r⟩
    arma::vec u = P_kpm.apply(e_r);
    arma::cx_vec u_cx(u, arma::vec(n_sites, arma::fill::zeros));

    // X|r⟩ (column r of X) and X|u⟩
    arma::cx_vec X_r = X.col(r);
    arma::cx_vec X_u = X * u_cx;

    // P(X|r⟩) and P(X|u⟩)
    arma::cx_vec P_X_r = P_kpm.apply(X_r);
    arma::cx_vec P_X_u = P_kpm.apply(X_u);

    // Combine: X u + P X r - 2 P X u
    arma::cx_vec combined = X_u + P_X_r - 2.0 * P_X_u;

    // W * combined (only need element r)
    std::complex<double> W_combined_r = 0.0;
    for (size_t i = 0; i < n_sites; ++i) {
      W_combined_r += W(r, i) * combined(i);
    }

    return std::imag(N1 * W_combined_r);
  }

  /// Build full marker operator matrix (expensive, for verification only)
  arma::cx_mat marker_operator() const {
    const std::complex<double> N1(0.0, 2.0 * std::numbers::pi);
    arma::cx_mat Wc = arma::conv_to<arma::cx_mat>::from(W);
    arma::cx_mat C(n_sites, n_sites);

    for (size_t r = 0; r < n_sites; ++r) {
      arma::vec e_r(n_sites, arma::fill::zeros);
      e_r(r) = 1.0;

      arma::vec u = P_kpm.apply(e_r);
      arma::cx_vec u_cx(u, arma::vec(n_sites, arma::fill::zeros));

      arma::cx_vec X_r = X.col(r);
      arma::cx_vec X_u = X * u_cx;

      arma::cx_vec P_X_r = P_kpm.apply(X_r);
      arma::cx_vec P_X_u = P_kpm.apply(X_u);

      arma::cx_vec combined = X_u + P_X_r - 2.0 * P_X_u;
      C.col(r) = N1 * (Wc * combined);
    }
    return C;
  }

  std::vector<double> local_marker() const {
    std::vector<double> marker(n_sites);
    for (size_t r = 0; r < n_sites; ++r) {
      marker[r] = local_marker_at(r);
    }
    return marker;
  }

  std::vector<double> local_marker_cells() const {
    const size_t num_cells = n_sites / 2;
    std::vector<double> marker(num_cells);
    const Index idx({2, num_cells});
    for (size_t n = 0; n < num_cells; ++n) {
      const size_t A = idx({0, n});
      const size_t B = idx({1, n});
      marker[n] = (local_marker_at(A) + local_marker_at(B)) / 2.0;
    }
    return marker;
  }

  double average_marker() const {
    double sum = 0.0;
    for (size_t r = 0; r < n_sites; ++r) {
      sum += local_marker_at(r);
    }
    return sum / static_cast<double>(n_sites);
  }

  double total_marker() const {
    double sum = 0.0;
    for (size_t r = 0; r < n_sites; ++r) {
      sum += local_marker_at(r);
    }
    return sum;
  }

  size_t expansion_order() const { return P_kpm.expansion_order(); }
};

}  // namespace topological
