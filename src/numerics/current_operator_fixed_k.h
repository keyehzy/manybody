#pragma once

#include <armadillo>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <numbers>
#include <stdexcept>
#include <vector>

#include "algebra/basis.h"
#include "utils/index.h"

/// Matrix-free current operator for fixed momentum sectors.
///
/// This operator implements the current operator in momentum space:
///
///   J_d(Q) = Σ_{k,σ} v_d(k) c†_{k+Q,σ} c_{k,σ}
///
/// where v_d(k) = 2t × (2π/L_d) × sin(2πk_d/L_d) is the group velocity.
///
/// The current operator maps from momentum sector K to sector K+Q:
///   J_d(Q): sector K → sector K+Q (shape: dim(K+Q) × dim(K))
///
/// This is a rectangular operator, so it provides both forward (apply) and
/// adjoint (adjoint_apply) methods rather than implementing LinearOperator.
struct CurrentOperatorFixedK {
  using VectorType = arma::cx_vec;
  using ScalarType = std::complex<double>;

  /// Construct the current operator for fixed momentum sectors.
  ///
  /// @param source_basis The many-body basis for sector K (source)
  /// @param target_basis The many-body basis for sector K+Q (target)
  /// @param lattice_size Lattice dimensions (e.g., {4, 4} for 2D 4x4 lattice)
  /// @param t Hopping amplitude
  /// @param Q Transfer momentum (flat index)
  /// @param direction Direction index d for the current (0, 1, or 2 for x, y, z)
  CurrentOperatorFixedK(const Basis& source_basis, const Basis& target_basis,
                        const std::vector<size_t>& lattice_size, double t, size_t Q,
                        size_t direction)
      : source_basis_(source_basis),
        target_basis_(target_basis),
        index_(lattice_size),
        t_(t),
        Q_(Q),
        direction_(direction),
        N_sites_(index_.size()) {
    if (lattice_size.empty()) {
      throw std::invalid_argument("CurrentOperatorFixedK requires at least one dimension.");
    }
    if (source_basis.orbitals != N_sites_ || target_basis.orbitals != N_sites_) {
      throw std::invalid_argument(
          "Basis orbitals must equal total number of lattice sites (momentum points).");
    }
    if (direction >= lattice_size.size()) {
      throw std::invalid_argument("Direction must be less than lattice dimensionality.");
    }

    precompute_momentum_shifts();
    precompute_velocities();
  }

  /// Source dimension (sector K).
  size_t source_dimension() const { return source_basis_.set.size(); }

  /// Target dimension (sector K+Q).
  size_t target_dimension() const { return target_basis_.set.size(); }

  /// Apply J_d(Q) to a vector: sector K → sector K+Q.
  VectorType apply(const VectorType& v) const {
    assert(static_cast<size_t>(v.n_elem) == source_dimension());

    VectorType w(target_dimension(), arma::fill::zeros);

    // For each source state
    for (size_t j = 0; j < source_basis_.set.size(); ++j) {
      if (std::abs(v(j)) < 1e-15) {
        continue;
      }

      const auto& state_j = source_basis_.set[j];

      // Apply current operator to this state
      apply_current_to_state(state_j, v(j), w, /* adjoint = */ false);
    }

    return w;
  }

  /// Apply J†_d(Q) to a vector: sector K+Q → sector K.
  VectorType adjoint_apply(const VectorType& v) const {
    assert(static_cast<size_t>(v.n_elem) == target_dimension());

    VectorType w(source_dimension(), arma::fill::zeros);

    // For each target state
    for (size_t j = 0; j < target_basis_.set.size(); ++j) {
      if (std::abs(v(j)) < 1e-15) {
        continue;
      }

      const auto& state_j = target_basis_.set[j];

      // Apply adjoint current operator to this state
      apply_adjoint_current_to_state(state_j, v(j), w);
    }

    return w;
  }

  /// Get the velocities (for testing/debugging).
  const std::vector<double>& velocities() const { return velocities_; }

 private:
  /// Precompute k+Q for all k.
  void precompute_momentum_shifts() {
    const auto& dims = index_.dimensions();
    const auto Q_coords = index_(Q_);

    k_plus_Q_.resize(N_sites_);
    for (size_t k = 0; k < N_sites_; ++k) {
      const auto k_coords = index_(k);
      std::vector<size_t> kQ_coords(dims.size());
      for (size_t d = 0; d < dims.size(); ++d) {
        kQ_coords[d] = (k_coords[d] + Q_coords[d]) % dims[d];
      }
      k_plus_Q_[k] = index_(kQ_coords);
    }
  }

  /// Precompute velocities v_d(k) = 2t × (2π/L_d) × sin(2πk_d/L_d) for all k.
  void precompute_velocities() {
    const auto& dims = index_.dimensions();

    velocities_.resize(N_sites_);
    for (size_t k = 0; k < N_sites_; ++k) {
      const auto k_coords = index_(k);
      const double phase = 2.0 * std::numbers::pi_v<double> *
                           static_cast<double>(k_coords[direction_]) /
                           static_cast<double>(dims[direction_]);
      velocities_[k] = 2.0 * t_ *
                       (2.0 * std::numbers::pi_v<double> / static_cast<double>(dims[direction_])) *
                       std::sin(phase);
    }
  }

  /// Apply J_d(Q) = Σ_{k,σ} v_d(k) c†_{k+Q,σ} c_{k,σ} to a basis state.
  void apply_current_to_state(const Basis::key_type& state_j, ScalarType coeff, VectorType& w,
                              bool /*adjoint*/) const {
    // For each occupied orbital k with spin σ
    for (size_t op_idx = 0; op_idx < state_j.size(); ++op_idx) {
      const auto& op = state_j[op_idx];
      const size_t k = op.value();
      const Operator::Spin spin = op.spin();

      const double v_k = velocities_[k];
      if (std::abs(v_k) < 1e-15) {
        continue;
      }

      const size_t kQ = k_plus_Q_[k];

      // Check if k+Q is already occupied with the same spin
      bool kQ_occupied = false;
      for (const auto& other_op : state_j) {
        if (other_op.value() == kQ && other_op.spin() == spin) {
          kQ_occupied = true;
          break;
        }
      }

      if (kQ_occupied && kQ != k) {
        // c†_{k+Q} c_k |state⟩ = 0 since k+Q is occupied (and k+Q != k)
        continue;
      }

      if (kQ == k) {
        // c†_k c_k = n_k, contributes v_k (diagonal term)
        if (target_basis_.set.contains(state_j)) {
          const size_t i = target_basis_.set.index_of(state_j);
          w(i) += coeff * ScalarType(v_k, 0.0);
        }
      } else {
        // Create the new state: remove operator at k, add operator at k+Q
        Operator new_op = Operator::creation(spin, kQ);

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

        // Look up the new state in the target basis
        if (target_basis_.set.contains(new_state)) {
          const size_t i = target_basis_.set.index_of(new_state);
          w(i) += coeff * ScalarType(static_cast<double>(sign) * v_k, 0.0);
        }
      }
    }
  }

  /// Apply J†_d(Q) = Σ_{k,σ} v_d(k)* c†_{k,σ} c_{k+Q,σ} to a basis state.
  /// Since v_d(k) is real, J†_d(Q) = Σ_{k,σ} v_d(k) c†_{k,σ} c_{k+Q,σ}.
  void apply_adjoint_current_to_state(const Basis::key_type& state_j, ScalarType coeff,
                                      VectorType& w) const {
    // For the adjoint, we need to find occupied k+Q and move to k.
    // J†_d(Q) = Σ_{k,σ} v_d(k) c†_{k,σ} c_{k+Q,σ}
    // This annihilates at k+Q and creates at k.

    // For each occupied orbital in the state
    for (size_t op_idx = 0; op_idx < state_j.size(); ++op_idx) {
      const auto& op = state_j[op_idx];
      const size_t kQ = op.value();  // This is k+Q in the adjoint
      const Operator::Spin spin = op.spin();

      // Find which k maps to this kQ
      // We need to find k such that k_plus_Q_[k] == kQ
      for (size_t k = 0; k < N_sites_; ++k) {
        if (k_plus_Q_[k] != kQ) {
          continue;
        }

        const double v_k = velocities_[k];
        if (std::abs(v_k) < 1e-15) {
          continue;
        }

        // Check if k is already occupied with the same spin
        bool k_occupied = false;
        for (const auto& other_op : state_j) {
          if (other_op.value() == k && other_op.spin() == spin) {
            k_occupied = true;
            break;
          }
        }

        if (k_occupied && k != kQ) {
          // c†_k c_{k+Q} |state⟩ = 0 since k is occupied
          continue;
        }

        if (k == kQ) {
          // c†_{k+Q} c_{k+Q} = n_{k+Q}, contributes v_k (diagonal term)
          if (source_basis_.set.contains(state_j)) {
            const size_t i = source_basis_.set.index_of(state_j);
            w(i) += coeff * ScalarType(v_k, 0.0);
          }
        } else {
          // Create the new state: remove operator at k+Q, add operator at k
          Operator new_op = Operator::creation(spin, k);

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

          // Look up the new state in the source basis
          if (source_basis_.set.contains(new_state)) {
            const size_t i = source_basis_.set.index_of(new_state);
            w(i) += coeff * ScalarType(static_cast<double>(sign) * v_k, 0.0);
          }
        }
      }
    }
  }

  const Basis& source_basis_;
  const Basis& target_basis_;
  Index index_;
  double t_;
  size_t Q_;
  size_t direction_;
  size_t N_sites_;

  std::vector<size_t> k_plus_Q_;    // k_plus_Q_[k] = k + Q
  std::vector<double> velocities_;  // velocities_[k] = v_d(k)
};
