#pragma once

#include <armadillo>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numbers>
#include <stdexcept>
#include <vector>

#include "numerics/linear_operator.h"
#include "utils/canonicalize_momentum.h"
#include "utils/index.h"

/// 3-particle relative-coordinate interaction for 2↑ + 1↓ Hubbard.
/// Coordinates are stacked as:
///   r2 = coords[0..D-1]    (↑ particle 2 relative to ↑ particle 1)
///   r3 = coords[D..2D-1]   (↓ particle relative to ↑ particle 1)
///
/// On-site U applies for ↓ overlapping either ↑:
///   (i)  r3 == 0
///   (ii) r3 == r2
struct HubbardRelative3Interaction final : LinearOperator<arma::cx_vec> {
  using VectorType = arma::cx_vec;
  using ScalarType = std::complex<double>;

  explicit HubbardRelative3Interaction(const std::vector<size_t>& size)
      : size_(size), dims_(size.size()), index_(make_relative_dims_(size)) {
    if (size_.empty()) {
      throw std::invalid_argument("HubbardRelative3Interaction requires at least one dimension.");
    }
  }

  size_t dimension() const override { return index_.size(); }

  VectorType apply(const VectorType& v) const override {
    assert(static_cast<size_t>(v.n_elem) == dimension());
    VectorType w(v.n_elem, arma::fill::zeros);

    for (size_t orbital = 0; orbital < dimension(); ++orbital) {
      const auto coords = index_(orbital);

      bool r3_is_zero = true;
      bool r3_equals_r2 = true;

      for (size_t d = 0; d < dims_; ++d) {
        const size_t r2d = coords[d];
        const size_t r3d = coords[dims_ + d];
        if (r3d != 0) r3_is_zero = false;
        if (r3d != r2d) r3_equals_r2 = false;
      }

      double n_overlap = 0.0;
      if (r3_is_zero) n_overlap += 1.0;
      if (r3_equals_r2) n_overlap += 1.0;

      w(orbital) = static_cast<ScalarType>(n_overlap) * v(orbital);
    }

    return w;
  }

  std::vector<size_t> size_;
  size_t dims_{0};
  Index index_;

 private:
  static std::vector<size_t> make_relative_dims_(const std::vector<size_t>& size) {
    // [Lx,Ly,...,Lx,Ly,...]
    std::vector<size_t> rel;
    rel.reserve(2 * size.size());
    rel.insert(rel.end(), size.begin(), size.end());
    rel.insert(rel.end(), size.begin(), size.end());
    return rel;
  }
};

/// 3-particle relative-coordinate kinetic operator in a total-momentum sector,
/// using the *reference gauge*:
/// - hops of particle 2 (↑) and particle 3 (↓) carry no COM phase
/// - hops of particle 1 (↑ reference) carry the full phase exp(± i k_d),
///   where k_d = 2π K_d / L_d (K_d integer, canonicalized mod L_d)
///
/// Basis coordinates: r2 = R2 - R1, r3 = R3 - R1 (mod L).
struct HubbardRelative3KineticReferenceGauge final : LinearOperator<arma::cx_vec> {
  using VectorType = arma::cx_vec;
  using ScalarType = std::complex<double>;

  HubbardRelative3KineticReferenceGauge(const std::vector<size_t>& size,
                                        const std::vector<int64_t>& total_momentum)
      : size_(size),
        dims_(size.size()),
        total_momentum_(utils::canonicalize_momentum(total_momentum, size)),
        index_(make_relative_dims_(size)) {
    if (size_.empty()) {
      throw std::invalid_argument(
          "HubbardRelative3KineticReferenceGauge requires at least one dimension.");
    }
    if (total_momentum.size() != dims_) {
      throw std::invalid_argument(
          "HubbardRelative3KineticReferenceGauge: size and momentum must match.");
    }

    k_phase_.resize(dims_);
    for (size_t d = 0; d < dims_; ++d) {
      k_phase_[d] = 2.0 * std::numbers::pi_v<double> * static_cast<double>(total_momentum_[d]) /
                    static_cast<double>(size_[d]);
    }
  }

  size_t dimension() const override { return index_.size(); }

  VectorType apply(const VectorType& v) const override {
    assert(static_cast<size_t>(v.n_elem) == dimension());
    VectorType w(v.n_elem, arma::fill::zeros);

    const size_t rel_dims = 2 * dims_;
    Index::offset_type offsets(rel_dims, 0);

    for (size_t orbital = 0; orbital < dimension(); ++orbital) {
      const auto coords = index_(orbital);

      for (size_t d = 0; d < dims_; ++d) {
        // -------------------------
        // Particle 2 (↑): r2_d hops
        // -------------------------
        offsets[d] = -1;
        w(orbital) += v(index_(coords, offsets));
        offsets[d] = +1;
        w(orbital) += v(index_(coords, offsets));
        offsets[d] = 0;

        // -------------------------
        // Particle 3 (↓): r3_d hops
        // -------------------------
        offsets[dims_ + d] = -1;
        w(orbital) += v(index_(coords, offsets));
        offsets[dims_ + d] = +1;
        w(orbital) += v(index_(coords, offsets));
        offsets[dims_ + d] = 0;

        // --------------------------------------------
        // Particle 1 (↑ reference): shifts r2 and r3
        // R1 -> R1 + e_d  => (r2,r3) -> (r2 - e_d, r3 - e_d)
        // In row-apply form:
        //   w(r2,r3) gets contributions from v(r2+e_d, r3+e_d) with phase e^{+ik_d}
        //   and from v(r2-e_d, r3-e_d) with phase e^{-ik_d}
        // --------------------------------------------
        const ScalarType phase_plus = std::exp(ScalarType(0.0, +k_phase_[d]));
        const ScalarType phase_minus = std::exp(ScalarType(0.0, -k_phase_[d]));

        offsets[d] = +1;
        offsets[dims_ + d] = +1;
        w(orbital) += phase_plus * v(index_(coords, offsets));

        offsets[d] = -1;
        offsets[dims_ + d] = -1;
        w(orbital) += phase_minus * v(index_(coords, offsets));

        offsets[d] = 0;
        offsets[dims_ + d] = 0;
      }
    }

    return w;
  }

  std::vector<size_t> size_;
  size_t dims_{0};
  std::vector<size_t> total_momentum_;
  Index index_;
  std::vector<double> k_phase_{};

 private:
  static std::vector<size_t> make_relative_dims_(const std::vector<size_t>& size) {
    std::vector<size_t> rel;
    rel.reserve(2 * size.size());
    rel.insert(rel.end(), size.begin(), size.end());
    rel.insert(rel.end(), size.begin(), size.end());
    return rel;
  }
};

/// Full 3-particle relative Hubbard operator (project convention):
///   H = t * K + U * V
struct HubbardRelative3ReferenceGauge final : LinearOperator<arma::cx_vec> {
  using VectorType = arma::cx_vec;

  HubbardRelative3ReferenceGauge(const std::vector<size_t>& size,
                                 const std::vector<int64_t>& total_momentum, double t, double U)
      : kinetic_(size, total_momentum), interaction_(size), t_(t), U_(U) {}

  size_t dimension() const override { return kinetic_.dimension(); }

  VectorType apply(const VectorType& v) const override {
    assert(static_cast<size_t>(v.n_elem) == dimension());
    return t_ * kinetic_.apply(v) + U_ * interaction_.apply(v);
  }

  HubbardRelative3KineticReferenceGauge kinetic_;
  HubbardRelative3Interaction interaction_;
  double t_{};
  double U_{};
};
