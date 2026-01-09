#pragma once

#include <armadillo>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <numbers>

#include "linear_operator.h"

struct HubbardRelativeKinetic final : LinearOperator<arma::vec> {
  using VectorType = arma::vec;
  using ScalarType = double;

  HubbardRelativeKinetic(size_t size, size_t total_momentum) : size_(size) {
    const double k_phase = 2.0 * std::numbers::pi_v<double> * static_cast<double>(total_momentum) /
                           static_cast<double>(size_);
    t_eff_ = 2.0 * std::cos(0.5 * k_phase);
  }

  size_t dimension() const override { return size_; }
  ScalarType effective_hopping() const { return t_eff_; }

  VectorType apply(const VectorType& v) const override {
    assert(v.n_elem == size_);
    VectorType w(v.n_elem, arma::fill::zeros);
    for (size_t r = 0; r < size_; ++r) {
      const size_t r_minus = (r + size_ - 1) % size_;
      const size_t r_plus = (r + 1) % size_;
      w(r) += t_eff_ * (v(r_minus) + v(r_plus));
    }
    return w;
  }

  size_t size_;
  ScalarType t_eff_{};
};

struct HubbardRelativeInteraction final : LinearOperator<arma::vec> {
  using VectorType = arma::vec;
  using ScalarType = double;

  explicit HubbardRelativeInteraction(size_t size) : size_(size) {}

  size_t dimension() const override { return size_; }

  VectorType apply(const VectorType& v) const override {
    assert(v.n_elem == size_);
    VectorType w(v.n_elem, arma::fill::zeros);
    w(0) = v(0);
    return w;
  }

  size_t size_;
};
