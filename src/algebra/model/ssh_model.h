#pragma once

#include <armadillo>
#include <cmath>
#include <cstddef>
#include <numbers>

#include "algebra/fermion/expression.h"
#include "algebra/model/model.h"
#include "utils/index.h"

/// SSH (Su-Schrieffer-Heeger) model for 1D topological insulator.
///
/// The SSH model describes a 1D chain with alternating hopping amplitudes:
///   - t1: intra-cell hopping (A to B within unit cell)
///   - t2: inter-cell hopping (B to A between adjacent cells)
///
/// Topological phases:
///   - t1 < t2: Topological phase (winding number = 1)
///   - t1 > t2: Trivial phase (winding number = 0)
///   - t1 = t2: Critical point (gap closes)
///
/// Site indexing uses Index({2, num_cells}) with coordinates {sublattice, cell}
/// where sublattice 0 = A, sublattice 1 = B
struct SSHModel : Model {
  static constexpr size_t SUBLATTICE_A = 0;
  static constexpr size_t SUBLATTICE_B = 1;

  SSHModel(double t1_val, double t2_val, size_t num_cells_val)
      : t1(t1_val),
        t2(t2_val),
        num_cells(num_cells_val),
        num_sites(2 * num_cells_val),
        index({2, num_cells_val}) {}

  /// Get site index for sublattice s in unit cell n
  size_t site(size_t sublattice, size_t cell) const { return index({sublattice, cell}); }

  /// Get A sublattice site index in unit cell n
  size_t site_A(size_t cell) const { return index({SUBLATTICE_A, cell}); }

  /// Get B sublattice site index in unit cell n
  size_t site_B(size_t cell) const { return index({SUBLATTICE_B, cell}); }

  /// Intra-cell hopping: A_n -> B_n with amplitude t1
  Expression intracell_hopping() const {
    Expression result;
    const auto coeff = Expression::complex_type(-t1, 0.0);
    for (size_t n = 0; n < num_cells; ++n) {
      result += coeff * hopping(site_A(n), site_B(n), Operator::Spin::Up);
    }
    return result;
  }

  /// Inter-cell hopping: B_n -> A_{n+1} with amplitude t2
  Expression intercell_hopping() const {
    Expression result;
    const auto coeff = Expression::complex_type(-t2, 0.0);
    for (size_t n = 0; n < num_cells; ++n) {
      const size_t next_cell = (n + 1) % num_cells;
      result += coeff * hopping(site_B(n), site_A(next_cell), Operator::Spin::Up);
    }
    return result;
  }

  Expression hamiltonian() const override { return intracell_hopping() + intercell_hopping(); }

  /// Build the single-particle Hamiltonian matrix directly.
  /// This is more efficient for non-interacting systems.
  /// Uses periodic boundary conditions.
  arma::mat single_particle_hamiltonian() const {
    arma::mat H(num_sites, num_sites, arma::fill::zeros);

    // Intra-cell hopping: A_n <-> B_n
    for (size_t n = 0; n < num_cells; ++n) {
      const size_t A = site_A(n);
      const size_t B = site_B(n);
      H(A, B) = -t1;
      H(B, A) = -t1;
    }

    // Inter-cell hopping: B_n <-> A_{n+1}
    for (size_t n = 0; n < num_cells; ++n) {
      const size_t B = site_B(n);
      const size_t A_next = site_A((n + 1) % num_cells);
      H(B, A_next) = -t2;
      H(A_next, B) = -t2;
    }

    return H;
  }

  /// Build the single-particle Hamiltonian with twisted boundary conditions.
  /// The boundary hopping B_{N-1} <-> A_0 gets a phase factor e^{iθ}.
  /// This breaks translation symmetry while preserving periodic topology.
  /// TBC is useful for computing topological invariants.
  arma::cx_mat single_particle_hamiltonian_tbc(double theta) const {
    arma::cx_mat H(num_sites, num_sites, arma::fill::zeros);
    const std::complex<double> phase = std::exp(std::complex<double>(0.0, theta));

    // Intra-cell hopping: A_n <-> B_n (no phase)
    for (size_t n = 0; n < num_cells; ++n) {
      const size_t A = site_A(n);
      const size_t B = site_B(n);
      H(A, B) = -t1;
      H(B, A) = -t1;
    }

    // Inter-cell hopping: B_n <-> A_{n+1}
    // Bulk hoppings (no phase)
    for (size_t n = 0; n + 1 < num_cells; ++n) {
      const size_t B = site_B(n);
      const size_t A_next = site_A(n + 1);
      H(B, A_next) = -t2;
      H(A_next, B) = -t2;
    }

    // Boundary hopping B_{N-1} <-> A_0 gets the twist phase
    const size_t B_last = site_B(num_cells - 1);
    const size_t A_first = site_A(0);
    H(B_last, A_first) = -t2 * phase;
    H(A_first, B_last) = -t2 * std::conj(phase);

    return H;
  }

  /// Build the single-particle Hamiltonian with open boundary conditions.
  /// This is important for observing edge states in the topological phase.
  arma::mat single_particle_hamiltonian_obc() const {
    arma::mat H(num_sites, num_sites, arma::fill::zeros);

    // Intra-cell hopping: A_n <-> B_n
    for (size_t n = 0; n < num_cells; ++n) {
      const size_t A = site_A(n);
      const size_t B = site_B(n);
      H(A, B) = -t1;
      H(B, A) = -t1;
    }

    // Inter-cell hopping: B_n <-> A_{n+1} (no wrapping)
    for (size_t n = 0; n + 1 < num_cells; ++n) {
      const size_t B = site_B(n);
      const size_t A_next = site_A(n + 1);
      H(B, A_next) = -t2;
      H(A_next, B) = -t2;
    }

    return H;
  }

  /// Compute the bulk gap analytically for the infinite system.
  /// Gap = 2 * |t1 - t2|
  double bulk_gap() const { return 2.0 * std::abs(t1 - t2); }

  /// Compute the correlation length (in units of lattice spacing).
  /// ξ = 1 / ln(max(t1,t2) / min(t1,t2))
  double correlation_length() const {
    if (t1 == t2) {
      return std::numeric_limits<double>::infinity();
    }
    const double ratio = std::max(t1, t2) / std::min(t1, t2);
    return 1.0 / std::log(ratio);
  }

  /// Momentum-space dispersion: E(k) = ±|t1 + t2 * exp(ik)|
  /// Returns {E_lower, E_upper} for given k
  std::pair<double, double> dispersion(double k) const {
    const double re = t1 + t2 * std::cos(k);
    const double im = t2 * std::sin(k);
    const double E = std::sqrt(re * re + im * im);
    return {-E, E};
  }

  /// Winding number (analytical for clean system)
  int winding_number() const {
    if (t1 < t2) {
      return 1;  // Topological
    }
    if (t1 > t2) {
      return 0;  // Trivial
    }
    return -1;  // Critical (undefined)
  }

  double t1;         // Intra-cell hopping
  double t2;         // Inter-cell hopping
  size_t num_cells;  // Number of unit cells
  size_t num_sites;  // Total number of sites (2 * num_cells)
  Index index;       // Index for {sublattice, cell} -> site mapping
};

/// Non-interacting SSH model utilities for single-particle calculations
namespace ssh {

/// Sublattice indices
constexpr size_t SUBLATTICE_A = 0;
constexpr size_t SUBLATTICE_B = 1;

/// Create an Index for SSH model with given number of cells
inline Index make_index(size_t num_cells) { return Index({2, num_cells}); }

/// Get site index for sublattice s in unit cell n
inline size_t site(const Index& idx, size_t sublattice, size_t cell) {
  return idx({sublattice, cell});
}

/// Get A sublattice site index in unit cell n
inline size_t site_A(const Index& idx, size_t cell) { return idx({SUBLATTICE_A, cell}); }

/// Get B sublattice site index in unit cell n
inline size_t site_B(const Index& idx, size_t cell) { return idx({SUBLATTICE_B, cell}); }

/// Build the projector onto occupied states (E < E_F)
/// For half-filling, E_F = 0
inline arma::mat build_projector(const arma::mat& H, double E_fermi = 0.0) {
  arma::vec eigvals;
  arma::mat eigvecs;
  arma::eig_sym(eigvals, eigvecs, H);

  arma::mat P(H.n_rows, H.n_cols, arma::fill::zeros);
  for (size_t i = 0; i < eigvals.n_elem; ++i) {
    if (eigvals(i) < E_fermi) {
      P += eigvecs.col(i) * eigvecs.col(i).t();
    }
  }
  return P;
}

/// Build the complementary projector Q = I - P
inline arma::mat build_complementary_projector(const arma::mat& P) {
  return arma::eye(P.n_rows, P.n_cols) - P;
}

/// Build the exponentiated position operator for PBC
/// x_hat = (L / 2π) * exp(2πi * x / L)
/// This cures edge anomalies in the marker (see Paper 2, Eq. 14)
inline arma::cx_mat build_position_operator_exp(size_t num_sites) {
  arma::cx_mat X(num_sites, num_sites, arma::fill::zeros);
  const std::complex<double> i(0.0, 1.0);
  const double L = static_cast<double>(num_sites);

  for (size_t j = 0; j < num_sites; ++j) {
    const double phase = 2.0 * std::numbers::pi * static_cast<double>(j) / L;
    X(j, j) = (L / (2.0 * std::numbers::pi)) * std::exp(i * phase);
  }
  return X;
}

/// Build the exponentiated position operator using unit-cell coordinates (for PBC).
/// A and B sites in the same unit cell share the same phase.
inline arma::cx_mat build_position_operator_exp_cells(size_t num_cells) {
  const size_t num_sites = 2 * num_cells;
  arma::cx_mat X(num_sites, num_sites, arma::fill::zeros);
  const std::complex<double> i(0.0, 1.0);
  const double L = static_cast<double>(num_cells);
  const Index idx = make_index(num_cells);

  for (size_t n = 0; n < num_cells; ++n) {
    const double phase = 2.0 * std::numbers::pi * static_cast<double>(n) / L;
    const std::complex<double> value = (L / (2.0 * std::numbers::pi)) * std::exp(i * phase);
    const size_t A = site_A(idx, n);
    const size_t B = site_B(idx, n);
    X(A, A) = value;
    X(B, B) = value;
  }
  return X;
}

/// Build the standard position operator (diagonal)
inline arma::mat build_position_operator(size_t num_sites) {
  arma::mat X(num_sites, num_sites, arma::fill::zeros);
  for (size_t j = 0; j < num_sites; ++j) {
    X(j, j) = static_cast<double>(j);
  }
  return X;
}

/// Build the position operator using unit-cell coordinates.
/// A and B sites in the same unit cell share the same position.
inline arma::mat build_position_operator_cells(size_t num_cells) {
  const size_t num_sites = 2 * num_cells;
  arma::mat X(num_sites, num_sites, arma::fill::zeros);
  const Index idx = make_index(num_cells);
  for (size_t n = 0; n < num_cells; ++n) {
    const double x = static_cast<double>(n);
    const size_t A = site_A(idx, n);
    const size_t B = site_B(idx, n);
    X(A, A) = x;
    X(B, B) = x;
  }
  return X;
}

/// Compute the chiral operator W = σ_z ⊗ I (sublattice symmetry)
/// For SSH: W_AA = +1, W_BB = -1
inline arma::mat build_chiral_operator(size_t num_cells) {
  const size_t num_sites = 2 * num_cells;
  arma::mat W(num_sites, num_sites, arma::fill::zeros);
  const Index idx = make_index(num_cells);
  for (size_t n = 0; n < num_cells; ++n) {
    const size_t A = site_A(idx, n);
    const size_t B = site_B(idx, n);
    W(A, A) = 1.0;   // A sublattice
    W(B, B) = -1.0;  // B sublattice
  }
  return W;
}

}  // namespace ssh
