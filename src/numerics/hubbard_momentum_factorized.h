#pragma once

#include <armadillo>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <numbers>
#include <stdexcept>
#include <vector>

#include "algebra/basis.h"
#include "numerics/linear_operator.h"
#include "utils/index.h"

/// Factorized Hubbard model in momentum space.
///
/// This operator implements the Hubbard Hamiltonian in momentum space using a factorized
/// representation of the interaction term that avoids dense matrix storage:
///
///   H = Σ_{k,σ} ε(k) n_{k,σ} + (U/N) Σ_q ρ_{q,↑} ρ_{-q,↓}
///
/// where:
///   - ε(k) = -2t Σ_d cos(2πk_d/L_d) is the dispersion
///   - ρ_{q,σ} = Σ_k c†_{k+q,σ} c_{k,σ} is the particle-hole density operator
///
/// The key insight is that each ρ_q is SPARSE in the many-body basis (O(N_particles) non-zeros
/// per column), so the interaction can be applied as a sum of sparse matrix products instead
/// of a single dense matrix.
///
/// Storage: O(N_sites × basis_size × N_particles) instead of O(basis_size²)
/// Apply cost: O(N_sites × basis_size × N_particles) instead of O(basis_size²)
struct HubbardMomentumFactorized final : LinearOperator<arma::cx_vec> {
  using VectorType = arma::cx_vec;
  using ScalarType = std::complex<double>;

  /// Construct the factorized Hubbard Hamiltonian.
  ///
  /// @param basis The many-body basis (should be with_fixed_particle_number_and_spin)
  /// @param lattice_size Lattice dimensions (e.g., {4, 4} for 2D 4x4 lattice)
  /// @param t Hopping amplitude
  /// @param U On-site interaction strength
  HubbardMomentumFactorized(const Basis& basis, const std::vector<size_t>& lattice_size, double t,
                            double U)
      : basis_(basis), index_(lattice_size), t_(t), U_(U), N_sites_(index_.size()) {
    if (lattice_size.empty()) {
      throw std::invalid_argument("HubbardMomentumFactorized requires at least one dimension.");
    }
    if (basis.orbitals != N_sites_) {
      throw std::invalid_argument(
          "Basis orbitals must equal total number of lattice sites (momentum points).");
    }

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

        // Apply ρ_{-q,↓} first, then ρ_{q,↑}
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

  /// Build the diagonal kinetic energy for each basis state.
  void build_kinetic_diagonal() {
    const size_t dim = basis_.set.size();
    kinetic_diagonal_.set_size(dim);

    for (size_t i = 0; i < dim; ++i) {
      const auto& state = basis_.set[i];
      double energy = 0.0;

      for (const auto& op : state) {
        const size_t k = op.value();
        energy += dispersion(k);
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
  void build_density_operators() {
    const size_t dim = basis_.set.size();
    const auto& dims = index_.dimensions();

    rho_up_.resize(N_sites_);
    rho_down_.resize(N_sites_);

    for (size_t q = 0; q < N_sites_; ++q) {
      rho_up_[q] = arma::sp_cx_mat(dim, dim);
      rho_down_[q] = arma::sp_cx_mat(dim, dim);
    }

    // For each basis state (column j), compute where ρ_{q,σ}|j⟩ lands
    for (size_t j = 0; j < dim; ++j) {
      const auto& state_j = basis_.set[j];

      // For each momentum transfer q
      for (size_t q = 0; q < N_sites_; ++q) {
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

        // Apply ρ_{q,↑} = Σ_k c†_{k+q,↑} c_{k,↑}
        apply_density_operator(state_j, j, q, k_plus_q, Operator::Spin::Up, rho_up_[q]);

        // Apply ρ_{q,↓} = Σ_k c†_{k+q,↓} c_{k,↓}
        apply_density_operator(state_j, j, q, k_plus_q, Operator::Spin::Down, rho_down_[q]);
      }
    }
  }

  /// Apply density operator ρ_{q,σ} = Σ_k c†_{k+q,σ} c_{k,σ} to a basis state.
  void apply_density_operator(const Basis::key_type& state_j, size_t j, size_t /*q*/,
                              const std::vector<size_t>& k_plus_q, Operator::Spin spin,
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
        rho_q(j, j) += ScalarType(1.0, 0.0);
      } else {
        // Create the new state: remove k at op_idx, add k+q at sorted position
        // We need to build the new state in sorted order

        Operator new_op = Operator::creation(spin, kq);

        // Find where new_op would be inserted in the sorted state (excluding op_idx)
        size_t insert_pos = 0;
        for (size_t i = 0; i < state_j.size(); ++i) {
          if (i == op_idx) continue;
          if (state_j[i] < new_op) {
            insert_pos++;
          }
        }

        // Build the new state in sorted order
        Basis::key_type new_state;
        size_t src_idx = 0;
        size_t dst_idx = 0;
        bool inserted = false;

        while (dst_idx < state_j.size()) {
          // Skip the removed operator
          if (src_idx == op_idx) {
            src_idx++;
            continue;
          }

          // Insert new_op at the correct position
          if (!inserted && dst_idx == insert_pos) {
            new_state.push_back(new_op);
            inserted = true;
            dst_idx++;
            continue;
          }

          // Copy from source
          if (src_idx < state_j.size()) {
            new_state.push_back(state_j[src_idx]);
            src_idx++;
            dst_idx++;
          }
        }

        // If new_op goes at the end
        if (!inserted) {
          new_state.push_back(new_op);
        }

        // Compute the fermionic sign
        // c†_{kq} c_k |state⟩ where state = c†_{i_1} ... c†_{i_n} |0⟩
        //
        // 1. c_k anticommutes with op_idx operators to reach position op_idx
        //    Sign: (-1)^{op_idx}
        //
        // 2. After annihilation, c†_{kq} is conceptually at position 0
        //    It needs to move to position insert_pos, anticommuting with insert_pos operators
        //    Sign: (-1)^{insert_pos}
        //
        // Total sign = (-1)^{op_idx + insert_pos}

        int sign = ((op_idx + insert_pos) % 2 == 0) ? 1 : -1;

        // Look up the new state in the basis
        if (basis_.set.contains(new_state)) {
          const size_t i = basis_.set.index_of(new_state);
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

  arma::vec kinetic_diagonal_;
  std::vector<arma::sp_cx_mat> rho_up_;
  std::vector<arma::sp_cx_mat> rho_down_;
};
