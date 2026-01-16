#pragma once

#include <armadillo>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <numbers>
#include <stdexcept>
#include <vector>

#include "numerics/linear_operator.h"
#include "utils/index.h"

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
    assert(v.n_elem == dimension());
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

  HubbardRelativeKinetic(const std::vector<size_t>& size, const std::vector<size_t>& total_momentum)
      : size_(size), total_momentum_(total_momentum), index_(size_) {
    if (size_.empty()) {
      throw std::invalid_argument("HubbardRelativeKinetic requires at least one dimension.");
    }
    if (size_.size() != total_momentum_.size()) {
      throw std::invalid_argument("HubbardRelativeKinetic size and momentum must match.");
    }
    t_eff_.resize(size_.size());
    for (size_t d = 0; d < size_.size(); ++d) {
      const double k_phase = 2.0 * std::numbers::pi_v<double> *
                             static_cast<double>(total_momentum_[d]) /
                             static_cast<double>(size_[d]);
      t_eff_[d] = 2.0 * std::cos(0.5 * k_phase);
    }
  }

  size_t dimension() const override { return index_.size(); }
  ScalarType effective_hopping(size_t dim) const {
    assert(dim < t_eff_.size());
    return t_eff_[dim];
  }

  VectorType apply(const VectorType& v) const override {
    assert(v.n_elem == dimension());
    VectorType w(v.n_elem, arma::fill::zeros);
    const size_t dims = size_.size();
    Index::offset_type offsets(dims, 0);
    for (size_t orbital = 0; orbital < dimension(); ++orbital) {
      const auto coords = index_(orbital);
      for (size_t d = 0; d < dims; ++d) {
        offsets[d] = -1;
        w(orbital) += t_eff_[d] * v(index_(coords, offsets));
        offsets[d] = 1;
        w(orbital) += t_eff_[d] * v(index_(coords, offsets));
        offsets[d] = 0;
      }
    }
    return w;
  }

  std::vector<size_t> size_;
  std::vector<size_t> total_momentum_;
  Index index_;
  std::vector<ScalarType> t_eff_{};
};

struct HubbardRelativeCurrent final : LinearOperator<arma::cx_vec> {
  using VectorType = arma::cx_vec;
  using ScalarType = std::complex<double>;

  HubbardRelativeCurrent(const std::vector<size_t>& size, const std::vector<size_t>& total_momentum,
                         double t, size_t direction)
      : size_(size), total_momentum_(total_momentum), index_(size_) {
    if (size_.empty()) {
      throw std::invalid_argument("HubbardRelativeCurrent requires at least one dimension.");
    }
    if (size_.size() != total_momentum_.size()) {
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
  }

  size_t dimension() const override { return index_.size(); }

  VectorType apply(const VectorType& v) const override {
    assert(v.n_elem == dimension());
    VectorType w(v.n_elem, arma::fill::zeros);

    Index::offset_type offsets(size_.size(), 0);
    for (size_t orbital = 0; orbital < dimension(); ++orbital) {
      const auto coords = index_(orbital);

      offsets[direction_] = -1;
      w(orbital) += current_coeff_ * v(index_(coords, offsets));
      offsets[direction_] = 1;
      w(orbital) += current_coeff_ * v(index_(coords, offsets));
      offsets[direction_] = 0;
    }

    return w;
  }

  std::vector<size_t> size_;
  std::vector<size_t> total_momentum_;
  ScalarType current_coeff_{};
  size_t direction_{0};
  Index index_;
};

struct CurrentRelative_Q final : LinearOperator<arma::cx_vec> {
  using VectorType = arma::cx_vec;
  using ScalarType = std::complex<double>;

  CurrentRelative_Q(const std::vector<size_t>& size, double t,
                    const std::vector<size_t>& total_momentum,
                    const std::vector<size_t>& transfer_momentum, size_t direction)
      : size_(size),
        t_(t),
        total_momentum_(total_momentum),
        transfer_momentum_(transfer_momentum),
        direction_(direction),
        index_(size_) {
    if (size_.empty()) {
      throw std::invalid_argument("CurrentRelative_Q requires at least one dimension.");
    }
    const size_t dims = size_.size();
    if (total_momentum_.size() != dims || transfer_momentum_.size() != dims) {
      throw std::invalid_argument(
          "CurrentRelative_Q: total and transfer momentum must match the number of dimensions.");
    }
    if (direction_ >= dims) {
      throw std::invalid_argument("CurrentRelative_Q: direction out of bounds.");
    }

    prefactor_ = 2.0 * t_;

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
    assert(static_cast<size_t>(v.n_elem) == dimension());
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
          static_cast<ScalarType>(-prefactor_ * std::sin((theta - hqd) - hKd));
      const ScalarType weight_right =
          static_cast<ScalarType>(prefactor_ * std::sin((theta + hqd) + hKd));

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
  double prefactor_{0};
};
