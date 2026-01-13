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

  HubbardRelativeKinetic(std::vector<size_t> size, std::vector<size_t> total_momentum)
      : size_(std::move(size)),
        total_momentum_(std::move(total_momentum)),
        index_(size_) {
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
    DynamicIndex::offset_type offsets(dims, 0);
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
  DynamicIndex index_;
  std::vector<ScalarType> t_eff_{};
};
