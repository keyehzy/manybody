#pragma once

#include <armadillo>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <numbers>
#include <stdexcept>
#include <vector>

#include "algebra/fermion/basis.h"
#include "numerics/linear_operator.h"
#include "utils/index.h"

/// Factorized Hubbard model in momentum space for a fixed total momentum sector.
///
/// This operator implements the Hubbard Hamiltonian in momentum space using a factorized
/// representation of the interaction term, specifically designed to work within a single
/// total momentum sector K:
///
///   H = Σ_{k,σ} ε(k) n_{k,σ} + (U/N) Σ_q ρ_{q,↑} ρ_{-q,↓}
///
/// where:
///   - ε(k) = -2t Σ_d cos(2πk_d/L_d) is the dispersion
///   - ρ_{q,σ} = Σ_k c†_{k+q,σ} c_{k,σ} is the particle-hole density operator
///
/// ## Why this class is needed
///
/// In a fixed total momentum sector K, the density operator ρ_{q,σ} changes total momentum
/// by +q, mapping states from sector K to sector K+q. For q≠0, this maps OUT of the original
/// sector. The standard HubbardMomentumFactorized class builds square matrices within a
/// single basis, which only works when the basis contains all momentum sectors.
///
/// This class handles the cross-sector mapping by building rectangular matrices:
///   - ρ_{q,σ}: sector K → sector K+q (shape: dim(K+q) × dim(K))
///
/// The interaction is computed as:
///   H_int|v⟩ = (U/N) Σ_q ρ_{q,↑}(K-q→K) · ρ_{-q,↓}(K→K-q) |v⟩
///
/// where the intermediate vector after ρ_{-q,↓} lives in sector K-q.
struct HubbardMomentumFactorizedFixedK final : LinearOperator<arma::cx_vec> {
  using VectorType = arma::cx_vec;
  using ScalarType = std::complex<double>;

  /// Construct the factorized Hubbard Hamiltonian for a fixed momentum sector.
  ///
  /// @param basis The many-body basis for sector K (from with_fixed_particle_number_spin_momentum)
  /// @param lattice_size Lattice dimensions (e.g., {4, 4} for 2D 4x4 lattice)
  /// @param t Hopping amplitude
  /// @param U On-site interaction strength
  /// @param target_momentum The total momentum K of the sector
  HubbardMomentumFactorizedFixedK(const Basis& basis, const std::vector<size_t>& lattice_size,
                                  double t, double U, const Index::container_type& target_momentum)
      : basis_(basis),
        index_(lattice_size),
        t_(t),
        U_(U),
        N_sites_(index_.size()),
        target_momentum_(target_momentum) {
    if (lattice_size.empty()) {
      throw std::invalid_argument(
          "HubbardMomentumFactorizedFixedK requires at least one dimension.");
    }
    if (basis.orbitals != N_sites_) {
      throw std::invalid_argument(
          "Basis orbitals must equal total number of lattice sites (momentum points).");
    }
    if (target_momentum.size() != lattice_size.size()) {
      throw std::invalid_argument(
          "Target momentum dimensionality must match lattice dimensionality.");
    }
    for (size_t d = 0; d < lattice_size.size(); ++d) {
      if (target_momentum[d] >= lattice_size[d]) {
        throw std::invalid_argument("Target momentum components must be less than lattice size.");
      }
    }

    build_sector_bases();
    build_kinetic_diagonal();
    build_density_operators();
  }

  size_t dimension() const override { return basis_.set.size(); }

  VectorType apply(const VectorType& v) const override {
    assert(static_cast<size_t>(v.n_elem) == dimension());

    // Start with kinetic term: H_kin |v⟩ = t × diag(ε) |v⟩
    VectorType w = t_ * (kinetic_diagonal_ % v);

    // Add interaction term: H_int |v⟩ = (U/N) Σ_q ρ_{q,↑} (ρ_{-q,↓} |v⟩)
    if (std::abs(U_) > 1e-15) {
      const double u_over_n = U_ / static_cast<double>(N_sites_);
      for (size_t q = 0; q < N_sites_; ++q) {
        // Compute -q in multi-dimensional momentum space
        const size_t minus_q = compute_minus_q(q);

        // Apply ρ_{-q,↓} first (maps K → K-q), then ρ_{q,↑} (maps K-q → K)
        // rho_down_[minus_q] = ρ_{-q,↓} maps K → K + (-q) = K - q
        // rho_up_[q] = ρ_{q,↑} maps K - q → K
        VectorType temp = rho_down_[minus_q] * v;
        w += u_over_n * (rho_up_[q] * temp);
      }
    }

    return w;
  }

  /// Get the kinetic diagonal (for testing/debugging).
  const arma::vec& kinetic_diagonal() const { return kinetic_diagonal_; }

  /// Get the density operator matrices (for testing/debugging).
  const std::vector<arma::sp_cx_mat>& rho_up() const { return rho_up_; }
  const std::vector<arma::sp_cx_mat>& rho_down() const { return rho_down_; }

  /// Get the sector bases (for testing/debugging).
  const std::vector<Basis>& sector_bases() const { return sector_bases_; }

 private:
  /// Compute the flat index for -q (component-wise negation with periodic BCs).
  size_t compute_minus_q(size_t q) const {
    const auto& dims = index_.dimensions();
    const auto q_coords = index_(q);
    std::vector<size_t> minus_q_coords(dims.size());
    for (size_t d = 0; d < dims.size(); ++d) {
      minus_q_coords[d] = (dims[d] - q_coords[d]) % dims[d];
    }
    return index_(minus_q_coords);
  }

  /// Compute K + q (component-wise addition with periodic BCs).
  Index::container_type add_momentum(const Index::container_type& k, size_t q) const {
    const auto& dims = index_.dimensions();
    const auto q_coords = index_(q);
    Index::container_type result(dims.size());
    for (size_t d = 0; d < dims.size(); ++d) {
      result[d] = (k[d] + q_coords[d]) % dims[d];
    }
    return result;
  }

  /// Build bases for all sectors K + q for q = 0, 1, ..., N_sites - 1.
  void build_sector_bases() {
    sector_bases_.resize(N_sites_);

    // Compute spin counts from the main basis
    size_t n_up = 0;
    size_t n_down = 0;
    if (!basis_.set.empty()) {
      const auto& first_state = basis_.set[0];
      for (const auto& op : first_state) {
        if (op.spin() == Operator::Spin::Up) {
          ++n_up;
        } else {
          ++n_down;
        }
      }
    }
    const int spin_projection = static_cast<int>(n_up) - static_cast<int>(n_down);

    for (size_t q = 0; q < N_sites_; ++q) {
      Index::container_type sector_momentum = add_momentum(target_momentum_, q);

      // Build basis for sector K + q
      sector_bases_[q] = Basis::with_fixed_particle_number_spin_momentum(
          basis_.orbitals, basis_.particles, spin_projection, index_, sector_momentum);
    }
  }

  /// Build the diagonal kinetic energy for each basis state.
  void build_kinetic_diagonal() {
    const size_t dim = basis_.set.size();
    kinetic_diagonal_.set_size(dim);

    for (size_t i = 0; i < dim; ++i) {
      const auto& state = basis_.set[i];
      double energy = 0.0;

      for (const auto& op : state) {
        energy += dispersion(op.value());
      }

      kinetic_diagonal_(i) = energy;
    }
  }

  /// Compute dispersion ε(k) = -2 Σ_d cos(2πk_d/L_d)
  double dispersion(size_t k_flat) const {
    const auto k_coords = index_(k_flat);
    const auto& dims = index_.dimensions();
    double energy = 0.0;

    for (size_t d = 0; d < dims.size(); ++d) {
      const double phase = 2.0 * std::numbers::pi_v<double> * static_cast<double>(k_coords[d]) /
                           static_cast<double>(dims[d]);
      energy += -2.0 * std::cos(phase);
    }

    return energy;
  }

  /// Build sparse density operator matrices ρ_{q,σ} for all q.
  /// These are rectangular matrices mapping between momentum sectors.
  void build_density_operators() {
    const size_t dim_K = basis_.set.size();
    const auto& dims = index_.dimensions();

    rho_up_.resize(N_sites_);
    rho_down_.resize(N_sites_);

    for (size_t q = 0; q < N_sites_; ++q) {
      // ρ_{q,↑} maps sector K-q → K
      // Source sector is K - q, which is sector_bases_[compute_minus_q(q)]
      const size_t minus_q = compute_minus_q(q);
      const size_t dim_source_up = sector_bases_[minus_q].set.size();
      rho_up_[q] = arma::sp_cx_mat(dim_K, dim_source_up);

      // ρ_{q,↓} maps sector K → K+q
      // Target sector is K + q, which is sector_bases_[q]
      const size_t dim_target_down = sector_bases_[q].set.size();
      rho_down_[q] = arma::sp_cx_mat(dim_target_down, dim_K);
    }

    // Build rho_up_[q]: maps sector K-q → K
    // For each q, source basis is sector_bases_[minus_q], target basis is basis_ (sector K)
    for (size_t q = 0; q < N_sites_; ++q) {
      const size_t minus_q = compute_minus_q(q);
      const Basis& source_basis = sector_bases_[minus_q];

      // Precompute k+q for all k
      std::vector<size_t> k_plus_q(N_sites_);
      for (size_t k = 0; k < N_sites_; ++k) {
        const auto k_coords = index_(k);
        std::vector<size_t> kq_coords(dims.size());
        for (size_t d = 0; d < dims.size(); ++d) {
          kq_coords[d] = (k_coords[d] + index_.value_at(q, d)) % dims[d];
        }
        k_plus_q[k] = index_(kq_coords);
      }

      // For each column j (state in source sector K-q)
      for (size_t j = 0; j < source_basis.set.size(); ++j) {
        const auto& state_j = source_basis.set[j];
        apply_density_operator_cross_sector(state_j, j, k_plus_q, Operator::Spin::Up, source_basis,
                                            basis_, rho_up_[q]);
      }
    }

    // Build rho_down_[q]: maps sector K → K+q
    // For each q, source basis is basis_ (sector K), target basis is sector_bases_[q]
    for (size_t q = 0; q < N_sites_; ++q) {
      const Basis& target_basis = sector_bases_[q];

      // Precompute k+q for all k
      std::vector<size_t> k_plus_q(N_sites_);
      for (size_t k = 0; k < N_sites_; ++k) {
        const auto k_coords = index_(k);
        std::vector<size_t> kq_coords(dims.size());
        for (size_t d = 0; d < dims.size(); ++d) {
          kq_coords[d] = (k_coords[d] + index_.value_at(q, d)) % dims[d];
        }
        k_plus_q[k] = index_(kq_coords);
      }

      // For each column j (state in source sector K)
      for (size_t j = 0; j < basis_.set.size(); ++j) {
        const auto& state_j = basis_.set[j];
        apply_density_operator_cross_sector(state_j, j, k_plus_q, Operator::Spin::Down, basis_,
                                            target_basis, rho_down_[q]);
      }
    }
  }

  /// Apply density operator ρ_{q,σ} = Σ_k c†_{k+q,σ} c_{k,σ} to a basis state,
  /// looking up the resulting state in a potentially different target basis.
  void apply_density_operator_cross_sector(const Basis::key_type& state_j, size_t j,
                                           const std::vector<size_t>& k_plus_q, Operator::Spin spin,
                                           const Basis& /*source_basis*/, const Basis& target_basis,
                                           arma::sp_cx_mat& rho_q) {
    // For each occupied orbital k with matching spin
    for (size_t op_idx = 0; op_idx < state_j.size(); ++op_idx) {
      const auto& op = state_j[op_idx];
      if (op.spin() != spin) {
        continue;
      }

      const size_t k = op.value();
      const size_t kq = k_plus_q[k];

      // Check if k+q is already occupied (would give zero due to Pauli exclusion)
      bool kq_occupied = false;
      for (const auto& other_op : state_j) {
        if (other_op.value() == kq && other_op.spin() == spin) {
          kq_occupied = true;
          break;
        }
      }

      if (kq_occupied && kq != k) {
        // c†_{k+q} c_k |state⟩ = 0 since k+q is occupied (and k+q != k)
        continue;
      }

      if (kq == k) {
        // c†_k c_k = n_k, contributes 1 (diagonal term)
        // This only happens when q=0, so source and target sectors are the same
        if (target_basis.set.contains(state_j)) {
          const size_t i = target_basis.set.index_of(state_j);
          rho_q(i, j) += ScalarType(1.0, 0.0);
        }
      } else {
        // Create the new state: copy, erase old operator, insert new one
        Operator new_op = Operator::creation(spin, kq);

        Basis::key_type new_state = state_j;
        new_state.erase(op_idx);

        // Find where new_op should be inserted to maintain sorted order
        size_t insert_pos = 0;
        while (insert_pos < new_state.size() && new_state[insert_pos] < new_op) {
          insert_pos++;
        }
        new_state.insert(insert_pos, new_op);

        // Compute the fermionic sign
        int sign = ((op_idx + insert_pos) % 2 == 0) ? 1 : -1;

        // Look up the new state in the TARGET basis
        if (target_basis.set.contains(new_state)) {
          const size_t i = target_basis.set.index_of(new_state);
          rho_q(i, j) += ScalarType(static_cast<double>(sign), 0.0);
        }
      }
    }
  }

  const Basis& basis_;
  Index index_;
  double t_;
  double U_;
  size_t N_sites_;
  Index::container_type target_momentum_;

  std::vector<Basis> sector_bases_;  // sector_bases_[q] = basis for sector K + q

  arma::vec kinetic_diagonal_;
  std::vector<arma::sp_cx_mat> rho_up_;    // rho_up_[q] maps K-q → K
  std::vector<arma::sp_cx_mat> rho_down_;  // rho_down_[q] maps K → K+q
};
