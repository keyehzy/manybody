#pragma once

#include <armadillo>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numbers>
#include <stdexcept>
#include <vector>

#include "numerics/linear_operator.h"
#include "utils/canonicalize_momentum.h"
#include "utils/index.h"

/// Two-particle relative coordinates use the symmetric gauge
///   Psi(x_up, x_down) = exp(i K . (x_up + x_down) / 2) phi(x_up - x_down),
/// in which every hop has a real amplitude but periodicity in x_up imposes the twisted boundary
/// condition phi(r + L_d e_d) = (-1)^{K_d} phi(r). A hop from `coord` by `offset` (+-1) along a
/// dimension of length `length` therefore picks up `boundary_sign` when it wraps around.
inline double relative_hop_sign(size_t coord, int offset, size_t length, double boundary_sign) {
  const bool wraps = (offset < 0 && coord == 0) || (offset > 0 && coord + 1 == length);
  return wraps ? boundary_sign : 1.0;
}

/// (-1)^{K_d} for each dimension, from canonicalized total momentum.
inline std::vector<double> relative_boundary_signs(const std::vector<size_t>& total_momentum) {
  std::vector<double> signs(total_momentum.size());
  for (size_t d = 0; d < total_momentum.size(); ++d) {
    signs[d] = (total_momentum[d] % 2 == 0) ? 1.0 : -1.0;
  }
  return signs;
}

struct HubbardRelativeInteraction final : LinearOperator<arma::cx_vec> {
  using VectorType = arma::cx_vec;
  using ScalarType = std::complex<double>;

  explicit HubbardRelativeInteraction(const std::vector<size_t>& size)
      : size_(size), index_(size_) {
    if (size_.empty()) {
      throw std::invalid_argument("HubbardRelativeInteraction requires at least one dimension.");
    }
  }

  size_t dimension() const override { return index_.size(); }

  VectorType apply(const VectorType& v) const override {
    require_vector_dimension(v, dimension(), "HubbardRelativeInteraction");
    VectorType w(v.n_elem, arma::fill::zeros);
    w(0) = v(0);
    return w;
  }

  std::vector<size_t> size_;
  Index index_;
};

struct HubbardRelativeKinetic final : LinearOperator<arma::cx_vec> {
  using VectorType = arma::cx_vec;
  using ScalarType = std::complex<double>;

  HubbardRelativeKinetic(const std::vector<size_t>& size,
                         const std::vector<int64_t>& total_momentum)
      : size_(size),
        total_momentum_(utils::canonicalize_momentum(total_momentum, size)),
        index_(size_) {
    if (size_.empty()) {
      throw std::invalid_argument("HubbardRelativeKinetic requires at least one dimension.");
    }
    if (size_.size() != total_momentum.size()) {
      throw std::invalid_argument("HubbardRelativeKinetic size and momentum must match.");
    }
    t_eff_.resize(size_.size());
    for (size_t d = 0; d < size_.size(); ++d) {
      const double k_phase = 2.0 * std::numbers::pi_v<double> *
                             static_cast<double>(total_momentum_[d]) /
                             static_cast<double>(size_[d]);
      t_eff_[d] = -2.0 * std::cos(0.5 * k_phase);
    }
    boundary_sign_ = relative_boundary_signs(total_momentum_);
  }

  size_t dimension() const override { return index_.size(); }
  ScalarType effective_hopping(size_t dim) const {
    if (dim >= t_eff_.size()) {
      throw std::out_of_range("HubbardRelativeKinetic::effective_hopping: dimension out of range.");
    }
    return t_eff_[dim];
  }

  VectorType apply(const VectorType& v) const override {
    require_vector_dimension(v, dimension(), "HubbardRelativeKinetic");
    VectorType w(v.n_elem, arma::fill::zeros);
    const size_t dims = size_.size();
    Index::offset_type offsets(dims, 0);
    for (size_t orbital = 0; orbital < dimension(); ++orbital) {
      const auto coords = index_(orbital);
      for (size_t d = 0; d < dims; ++d) {
        for (const int step : {-1, 1}) {
          offsets[d] = step;
          const double sign = relative_hop_sign(coords[d], step, size_[d], boundary_sign_[d]);
          w(orbital) += sign * t_eff_[d] * v(index_(coords, offsets));
        }
        offsets[d] = 0;
      }
    }
    return w;
  }

  std::vector<size_t> size_;
  std::vector<size_t> total_momentum_;
  Index index_;
  std::vector<ScalarType> t_eff_{};
  std::vector<double> boundary_sign_{};
};

struct HubbardRelative final : LinearOperator<arma::cx_vec> {
  using VectorType = arma::cx_vec;
  using ScalarType = std::complex<double>;

  HubbardRelative(const std::vector<size_t>& size, const std::vector<int64_t>& total_momentum,
                  double t, double U)
      : kinetic_(size, total_momentum), interaction_(size), t_(t), U_(U) {}

  size_t dimension() const override { return kinetic_.dimension(); }

  VectorType apply(const VectorType& v) const override {
    require_vector_dimension(v, dimension(), "HubbardRelative");
    return t_ * kinetic_.apply(v) + U_ * interaction_.apply(v);
  }

  HubbardRelativeKinetic kinetic_;
  HubbardRelativeInteraction interaction_;
  double t_{};
  double U_{};
};

struct HubbardRelativeCurrent final : LinearOperator<arma::cx_vec> {
  using VectorType = arma::cx_vec;
  using ScalarType = std::complex<double>;

  HubbardRelativeCurrent(const std::vector<size_t>& size,
                         const std::vector<int64_t>& total_momentum, double t, size_t direction)
      : size_(size),
        total_momentum_(utils::canonicalize_momentum(total_momentum, size)),
        index_(size_) {
    if (size_.empty()) {
      throw std::invalid_argument("HubbardRelativeCurrent requires at least one dimension.");
    }
    if (size_.size() != total_momentum.size()) {
      throw std::invalid_argument("HubbardRelativeCurrent size and momentum must match.");
    }
    if (direction >= size_.size()) {
      throw std::invalid_argument("HubbardRelativeCurrent direction out of bounds.");
    }
    const double k_phase = 2.0 * std::numbers::pi_v<double> *
                           static_cast<double>(total_momentum_[direction]) /
                           static_cast<double>(size_[direction]);
    current_coeff_ = ScalarType(2.0 * t * std::sin(0.5 * k_phase));
    direction_ = direction;
    boundary_sign_ = relative_boundary_signs(total_momentum_)[direction];
  }

  size_t dimension() const override { return index_.size(); }

  VectorType apply(const VectorType& v) const override {
    require_vector_dimension(v, dimension(), "HubbardRelativeCurrent");
    VectorType w(v.n_elem, arma::fill::zeros);

    Index::offset_type offsets(size_.size(), 0);
    for (size_t orbital = 0; orbital < dimension(); ++orbital) {
      const auto coords = index_(orbital);

      for (const int step : {-1, 1}) {
        offsets[direction_] = step;
        const double sign =
            relative_hop_sign(coords[direction_], step, size_[direction_], boundary_sign_);
        w(orbital) += sign * current_coeff_ * v(index_(coords, offsets));
      }
      offsets[direction_] = 0;
    }

    return w;
  }

  std::vector<size_t> size_;
  std::vector<size_t> total_momentum_;
  ScalarType current_coeff_{};
  size_t direction_{0};
  double boundary_sign_{1.0};
  Index index_;
};

struct CurrentRelative_Q final : LinearOperator<arma::cx_vec> {
  using VectorType = arma::cx_vec;
  using ScalarType = std::complex<double>;

  CurrentRelative_Q(const std::vector<size_t>& size, double t,
                    const std::vector<int64_t>& total_momentum,
                    const std::vector<int64_t>& transfer_momentum, size_t direction)
      : size_(size),
        t_(t),
        total_momentum_(utils::canonicalize_momentum(total_momentum, size)),
        transfer_momentum_(utils::canonicalize_momentum(transfer_momentum, size)),
        direction_(direction),
        index_(size_),
        half_total_momentum_{},
        half_transfer_momentum_{} {
    if (size_.empty()) {
      throw std::invalid_argument("CurrentRelative_Q requires at least one dimension.");
    }
    const size_t dims = size_.size();
    if (total_momentum.size() != dims || transfer_momentum.size() != dims) {
      throw std::invalid_argument(
          "CurrentRelative_Q: total and transfer momentum must match the number of dimensions.");
    }
    if (direction_ >= dims) {
      throw std::invalid_argument("CurrentRelative_Q: direction out of bounds.");
    }

    half_total_momentum_.resize(dims);
    half_transfer_momentum_.resize(dims);
    for (size_t d = 0; d < dims; ++d) {
      const double total_phase = 2.0 * std::numbers::pi_v<double> *
                                 static_cast<double>(total_momentum_[d]) /
                                 static_cast<double>(size_[d]);
      const double transfer_phase = 2.0 * std::numbers::pi_v<double> *
                                    static_cast<double>(transfer_momentum_[d]) /
                                    static_cast<double>(size_[d]);
      half_total_momentum_[d] = total_phase / 2.0;
      half_transfer_momentum_[d] = transfer_phase / 2.0;
    }
  }

  size_t dimension() const override { return index_.size(); }

  VectorType apply(const VectorType& v) const override {
    require_vector_dimension(v, dimension(), "CurrentRelative_Q");
    VectorType w(v.n_elem, arma::fill::zeros);

    const size_t dims = size_.size();
    Index::offset_type offsets(dims, 0);

    const double hKd = half_total_momentum_[direction_];
    const double hqd = half_transfer_momentum_[direction_];

    for (size_t orbital = 0; orbital < dimension(); ++orbital) {
      const auto coords = index_(orbital);

      double theta = 0.0;
      for (size_t d = 0; d < dims; ++d) {
        theta += half_transfer_momentum_[d] * static_cast<double>(coords[d]);
      }

      offsets[direction_] = -1;
      const auto j_minus = index_(coords, offsets);
      offsets[direction_] = 1;
      const auto j_plus = index_(coords, offsets);
      offsets[direction_] = 0;

      const ScalarType weight_left =
          static_cast<ScalarType>(-2.0 * t_ * std::sin((theta - hqd) - hKd));
      const ScalarType weight_right =
          static_cast<ScalarType>(2.0 * t_ * std::sin((theta + hqd) + hKd));

      w(orbital) += weight_left * v(j_minus);
      w(orbital) += weight_right * v(j_plus);
    }

    return w;
  }

  std::vector<size_t> size_;
  double t_{};
  std::vector<size_t> total_momentum_;
  std::vector<size_t> transfer_momentum_;
  size_t direction_{0};

  Index index_;
  std::vector<double> half_total_momentum_;
  std::vector<double> half_transfer_momentum_;
};

/// Adjoint of CurrentRelative_Q: computes J(q)† correctly.
/// For the original J(q), matrix elements are:
///   J_{r,r-1} = -2t·sin(θ_r - q/2 - K/2)
///   J_{r,r+1} = 2t·sin(θ_r + q/2 + K/2)
/// where θ_r = q·r·π/L
///
/// For the adjoint J†, we have (J†)_{r,s} = conj(J_{s,r}):
///   (J†)_{r,r-1} = conj(J_{r-1,r}) = 2t·sin(θ_{r-1} + q/2 + K/2)
///   (J†)_{r,r+1} = conj(J_{r+1,r}) = -2t·sin(θ_{r+1} - q/2 - K/2)
struct CurrentRelative_Q_Adjoint final : LinearOperator<arma::cx_vec> {
  using VectorType = arma::cx_vec;
  using ScalarType = std::complex<double>;

  CurrentRelative_Q_Adjoint(const std::vector<size_t>& size, double t,
                            const std::vector<int64_t>& total_momentum,
                            const std::vector<int64_t>& transfer_momentum, size_t direction)
      : size_(size),
        t_(t),
        total_momentum_(utils::canonicalize_momentum(total_momentum, size)),
        transfer_momentum_(utils::canonicalize_momentum(transfer_momentum, size)),
        direction_(direction),
        index_(size_),
        half_total_momentum_{},
        half_transfer_momentum_{} {
    if (size_.empty()) {
      throw std::invalid_argument("CurrentRelative_Q_Adjoint requires at least one dimension.");
    }
    const size_t dims = size_.size();
    if (total_momentum.size() != dims || transfer_momentum.size() != dims) {
      throw std::invalid_argument(
          "CurrentRelative_Q_Adjoint: total and transfer momentum must match the number of "
          "dimensions.");
    }
    if (direction_ >= dims) {
      throw std::invalid_argument("CurrentRelative_Q_Adjoint: direction out of bounds.");
    }

    half_total_momentum_.resize(dims);
    half_transfer_momentum_.resize(dims);
    for (size_t d = 0; d < dims; ++d) {
      const double total_phase = 2.0 * std::numbers::pi_v<double> *
                                 static_cast<double>(total_momentum_[d]) /
                                 static_cast<double>(size_[d]);
      const double transfer_phase = 2.0 * std::numbers::pi_v<double> *
                                    static_cast<double>(transfer_momentum_[d]) /
                                    static_cast<double>(size_[d]);
      half_total_momentum_[d] = total_phase / 2.0;
      half_transfer_momentum_[d] = transfer_phase / 2.0;
    }
  }

  size_t dimension() const override { return index_.size(); }

  VectorType apply(const VectorType& v) const override {
    require_vector_dimension(v, dimension(), "CurrentRelative_Q_Adjoint");
    VectorType w(v.n_elem, arma::fill::zeros);

    const size_t dims = size_.size();
    Index::offset_type offsets(dims, 0);

    const double hKd = half_total_momentum_[direction_];
    const double hqd = half_transfer_momentum_[direction_];

    for (size_t orbital = 0; orbital < dimension(); ++orbital) {
      const auto coords = index_(orbital);

      // Get neighbor indices
      offsets[direction_] = -1;
      const auto j_minus = index_(coords, offsets);
      offsets[direction_] = 1;
      const auto j_plus = index_(coords, offsets);
      offsets[direction_] = 0;

      // Compute theta at neighbor positions (not at orbital position!)
      const auto coords_minus = index_(j_minus);
      const auto coords_plus = index_(j_plus);

      double theta_minus = 0.0;
      double theta_plus = 0.0;
      for (size_t d = 0; d < dims; ++d) {
        theta_minus += half_transfer_momentum_[d] * static_cast<double>(coords_minus[d]);
        theta_plus += half_transfer_momentum_[d] * static_cast<double>(coords_plus[d]);
      }

      // For adjoint: (J†)_{r,r-1} = conj(J_{r-1,r}) = 2t·sin(θ_{r-1} + hqd + hKd)
      // This is the "right" formula evaluated at the left neighbor
      const ScalarType weight_left =
          static_cast<ScalarType>(2.0 * t_ * std::sin(theta_minus + hqd + hKd));

      // For adjoint: (J†)_{r,r+1} = conj(J_{r+1,r}) = -2t·sin(θ_{r+1} - hqd - hKd)
      // This is the "left" formula evaluated at the right neighbor
      const ScalarType weight_right =
          static_cast<ScalarType>(-2.0 * t_ * std::sin(theta_plus - hqd - hKd));

      w(orbital) += weight_left * v(j_minus);
      w(orbital) += weight_right * v(j_plus);
    }

    return w;
  }

  std::vector<size_t> size_;
  double t_{};
  std::vector<size_t> total_momentum_;
  std::vector<size_t> transfer_momentum_;
  size_t direction_{0};

  Index index_;
  std::vector<double> half_total_momentum_;
  std::vector<double> half_transfer_momentum_;
};
