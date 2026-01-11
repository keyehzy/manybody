#pragma once

#include <armadillo>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <numbers>

#include "index.h"
#include "linear_operator.h"

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

struct HubbardRelativeKinetic final : LinearOperator<arma::vec> {
  using VectorType = arma::vec;
  using ScalarType = double;

  HubbardRelativeKinetic(size_t size, size_t total_momentum) : size_(size), index_({size}) {
    const double k_phase = 2.0 * std::numbers::pi_v<double> * static_cast<double>(total_momentum) /
                           static_cast<double>(size_);
    t_eff_ = 2.0 * std::cos(0.5 * k_phase);
  }

  size_t dimension() const override { return size_; }
  ScalarType effective_hopping() const { return t_eff_; }

  VectorType apply(const VectorType& v) const override {
    assert(v.n_elem == dimension());
    VectorType w(v.n_elem, arma::fill::zeros);
    for (size_t r = 0; r < size_; ++r) {
      const size_t i = index_({r});
      w(i) += t_eff_ * (v(index_({r}, {-1})) + v(index_({r}, {1})));
    }
    return w;
  }

  size_t size_;
  DynamicIndex index_;
  ScalarType t_eff_{};
};

struct HubbardRelativeKinetic2D final : LinearOperator<arma::vec> {
  using VectorType = arma::vec;
  using ScalarType = double;

  HubbardRelativeKinetic2D(size_t size, size_t total_momentum_x, size_t total_momentum_y)
      : size_(size),
        total_momentum_x_(total_momentum_x),
        total_momentum_y_(total_momentum_y),
        index_({size, size}) {
    const double kx_phase = 2.0 * std::numbers::pi_v<double> *
                            static_cast<double>(total_momentum_x_) / static_cast<double>(size_);
    const double ky_phase = 2.0 * std::numbers::pi_v<double> *
                            static_cast<double>(total_momentum_y_) / static_cast<double>(size_);
    tx_eff_ = 2.0 * std::cos(0.5 * kx_phase);
    ty_eff_ = 2.0 * std::cos(0.5 * ky_phase);
  }

  size_t dimension() const override { return size_ * size_; }
  ScalarType effective_hopping_x() const { return tx_eff_; }
  ScalarType effective_hopping_y() const { return ty_eff_; }

  VectorType apply(const VectorType& v) const override {
    assert(v.n_elem == dimension());
    VectorType w(v.n_elem, arma::fill::zeros);
    for (size_t y = 0; y < size_; ++y) {
      for (size_t x = 0; x < size_; ++x) {
        const size_t i = index_({x, y});
        w(i) += tx_eff_ * (v(index_({x, y}, {-1, 0})) + v(index_({x, y}, {1, 0})));
        w(i) += ty_eff_ * (v(index_({x, y}, {0, -1})) + v(index_({x, y}, {0, 1})));
      }
    }
    return w;
  }

  size_t size_;
  size_t total_momentum_x_;
  size_t total_momentum_y_;
  DynamicIndex index_;
  ScalarType tx_eff_{};
  ScalarType ty_eff_{};
};

struct HubbardRelativeKinetic3D final : LinearOperator<arma::vec> {
  using VectorType = arma::vec;
  using ScalarType = double;

  HubbardRelativeKinetic3D(size_t size, size_t total_momentum_x, size_t total_momentum_y,
                           size_t total_momentum_z)
      : size_(size),
        total_momentum_x_(total_momentum_x),
        total_momentum_y_(total_momentum_y),
        total_momentum_z_(total_momentum_z),
        index_({size, size, size}) {
    const double kx_phase = 2.0 * std::numbers::pi_v<double> *
                            static_cast<double>(total_momentum_x_) / static_cast<double>(size_);
    const double ky_phase = 2.0 * std::numbers::pi_v<double> *
                            static_cast<double>(total_momentum_y_) / static_cast<double>(size_);
    const double kz_phase = 2.0 * std::numbers::pi_v<double> *
                            static_cast<double>(total_momentum_z_) / static_cast<double>(size_);
    tx_eff_ = 2.0 * std::cos(0.5 * kx_phase);
    ty_eff_ = 2.0 * std::cos(0.5 * ky_phase);
    tz_eff_ = 2.0 * std::cos(0.5 * kz_phase);
  }

  size_t dimension() const override { return size_ * size_ * size_; }
  ScalarType effective_hopping_x() const { return tx_eff_; }
  ScalarType effective_hopping_y() const { return ty_eff_; }
  ScalarType effective_hopping_z() const { return tz_eff_; }

  VectorType apply(const VectorType& v) const override {
    assert(v.n_elem == dimension());
    VectorType w(v.n_elem, arma::fill::zeros);
    for (size_t z = 0; z < size_; ++z) {
      for (size_t y = 0; y < size_; ++y) {
        for (size_t x = 0; x < size_; ++x) {
          const size_t i = index_({x, y, z});
          w(i) += tx_eff_ * (v(index_({x, y, z}, {-1, 0, 0})) + v(index_({x, y, z}, {1, 0, 0})));
          w(i) += ty_eff_ * (v(index_({x, y, z}, {0, -1, 0})) + v(index_({x, y, z}, {0, 1, 0})));
          w(i) += tz_eff_ * (v(index_({x, y, z}, {0, 0, -1})) + v(index_({x, y, z}, {0, 0, 1})));
        }
      }
    }
    return w;
  }

  size_t size_;
  size_t total_momentum_x_;
  size_t total_momentum_y_;
  size_t total_momentum_z_;
  DynamicIndex index_;
  ScalarType tx_eff_{};
  ScalarType ty_eff_{};
  ScalarType tz_eff_{};
};
