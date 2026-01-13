#pragma once

#include <armadillo>
#include <cmath>
#include <complex>
#include <limits>

#include "numerics/linear_operator.h"

template <typename Scalar>
struct Bounds {
  using RealType = scalar_real_t<Scalar>;

  RealType alpha;
  RealType beta;
};

template <typename OperatorType>
auto power_method(const OperatorType& op, typename OperatorType::VectorType v, size_t iterations) {
  using ScalarType = typename OperatorType::ScalarType;
  using RealType = scalar_real_t<ScalarType>;

  if (iterations == 0) {
    return static_cast<RealType>(0);
  }

  RealType norm = arma::norm(v);
  if (norm == static_cast<RealType>(0)) {
    return static_cast<RealType>(0);
  }
  v /= static_cast<ScalarType>(norm);

  RealType eigenvalue = static_cast<RealType>(0);
  for (size_t i = 0; i < iterations; ++i) {
    auto w = op.apply(v);
    const auto denom = arma::dot(v, v);
    const auto numerator = arma::dot(v, w);
    const RealType denom_real = std::real(denom);
    if (denom_real == static_cast<RealType>(0)) {
      return static_cast<RealType>(0);
    }
    eigenvalue = std::real(numerator) / denom_real;

    norm = arma::norm(w);
    if (norm == static_cast<RealType>(0)) {
      return static_cast<RealType>(0);
    }
    v = w / static_cast<ScalarType>(norm);
  }
  return eigenvalue;
}

template <typename Op>
Bounds<typename Op::ScalarType> estimate_bounds(
    const Op& op, size_t power_iterations,
    scalar_real_t<typename Op::ScalarType> spectral_padding) {
  using ScalarType = typename Op::ScalarType;
  using RealType = scalar_real_t<ScalarType>;
  const auto seed = make_seed_vector(op);

  const RealType dominant = std::abs(power_method(op, seed, power_iterations));
  if (dominant == static_cast<RealType>(0)) {
    return {static_cast<RealType>(0), static_cast<RealType>(0)};
  }

  const RealType shift = dominant * (static_cast<RealType>(1) + spectral_padding) +
                         std::numeric_limits<RealType>::epsilon();

  Shifted<Op> shifted(op, static_cast<ScalarType>(shift));
  NegShift<Op> neg_shifted(op, static_cast<ScalarType>(shift));

  const RealType beta = power_method(shifted, seed, power_iterations) - shift;
  const RealType alpha = shift - power_method(neg_shifted, seed, power_iterations);
  if (alpha <= beta) {
    return {alpha, beta};
  }
  return {beta, alpha};
}
