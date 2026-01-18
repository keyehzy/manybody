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

/// Swap the two ↑ particles in reference gauge:
///   (r2, r3) -> (-r2, r3 - r2) mod L.
inline std::vector<size_t> exchange_coords(const std::vector<size_t>& coords,
                                           const std::vector<size_t>& size) {
  const size_t dims = size.size();
  std::vector<size_t> out(coords);

  // coords = [r2(0..D-1), r3(0..D-1)]
  for (size_t d = 0; d < dims; ++d) {
    const size_t L = size[d];
    const size_t r2 = coords[d];
    const size_t r3 = coords[dims + d];

    const size_t r2p = (L - (r2 % L)) % L;
    const size_t r3p = (r3 + L - (r2 % L)) % L;

    out[d] = r2p;
    out[dims + d] = r3p;
  }
  return out;
}

// (P12 v)(r2,r3) = exp(+i K·r2) v(-r2, r3-r2)
inline std::complex<double> exchange_phase(const std::vector<size_t>& coords,
                                           const std::vector<size_t>& size,
                                           const std::vector<size_t>& K_canon) {
  const size_t dims = size.size();
  double theta = 0.0;
  for (size_t d = 0; d < dims; ++d) {
    const double kd = 2.0 * std::numbers::pi_v<double> * static_cast<double>(K_canon[d]) /
                      static_cast<double>(size[d]);
    theta += kd * static_cast<double>(coords[d]);  // r2_d
  }
  return std::exp(std::complex<double>(0.0, theta));
}

inline std::vector<size_t> make_relative_dims(const std::vector<size_t>& size) {
  std::vector<size_t> rel;
  rel.reserve(2 * size.size());
  rel.insert(rel.end(), size.begin(), size.end());
  rel.insert(rel.end(), size.begin(), size.end());
  return rel;
}

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
      : size_(size), dims_(size.size()), index_(make_relative_dims(size)) {}

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
};

/// 3-particle relative-coordinate kinetic operator in a total-momentum sector,
/// using the *reference gauge*:
/// - hops of particle 2 (↑) and particle 3 (↓) carry no COM phase
/// - hops of particle 1 (↑ reference) carry the full phase exp(± i k_d),
///   where k_d = 2π K_d / L_d (K_d integer, canonicalized mod L_d)
///
/// Basis coordinates: r2 = R2 - R1, r3 = R3 - R1 (mod L).
struct HubbardRelative3Kinetic final : LinearOperator<arma::cx_vec> {
  using VectorType = arma::cx_vec;
  using ScalarType = std::complex<double>;

  HubbardRelative3Kinetic(const std::vector<size_t>& size,
                          const std::vector<int64_t>& total_momentum)
      : size_(size),
        dims_(size.size()),
        total_momentum_(utils::canonicalize_momentum(total_momentum, size)),
        index_(make_relative_dims(size)) {
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
};

/// Full 3-particle relative Hubbard operator (project convention):
///   H = t * K + U * V
struct HubbardRelative3 final : LinearOperator<arma::cx_vec> {
  using VectorType = arma::cx_vec;
  using ScalarType = std::complex<double>;

  HubbardRelative3(const std::vector<size_t>& size, const std::vector<int64_t>& total_momentum,
                   double t, double U)
      : size_(size),
        K_canon_(utils::canonicalize_momentum(total_momentum, size)),
        index_(make_relative_dims(size)),
        kinetic_(size, total_momentum),
        interaction_(size),
        t_(t),
        U_(U) {}

  size_t dimension() const override { return kinetic_.dimension(); }

  VectorType apply(const VectorType& v) const override {
    assert(static_cast<size_t>(v.n_elem) == dimension());
    return t_ * kinetic_.apply(v) + U_ * interaction_.apply(v);
  }

  VectorType project_antisymmetric(const VectorType& v) const {
    VectorType w(v.n_elem, arma::fill::zeros);

    for (size_t i = 0; i < static_cast<size_t>(v.n_elem); ++i) {
      const auto coords = index_(i);
      const auto partner_coords = exchange_coords(coords, size_);
      const size_t j = index_(partner_coords);

      const ScalarType phase = exchange_phase(coords, size_, K_canon_);
      w(i) = 0.5 * (v(i) - phase * v(j));
    }
    return w;
  }

  std::vector<size_t> size_;
  std::vector<size_t> K_canon_;
  Index index_;
  HubbardRelative3Kinetic kinetic_;
  HubbardRelative3Interaction interaction_;
  double t_{};
  double U_{};
};
