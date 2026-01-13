#pragma once

#include <algorithm>
#include <armadillo>
#include <cassert>
#include <complex>
#include <cstddef>
#include <limits>
#include <type_traits>
#include <utility>
#include <vector>

#include "special_functions.h"

template <typename T>
struct ScalarRealType {
  using type = T;
};

template <typename T>
struct ScalarRealType<std::complex<T>> {
  using type = T;
};

template <typename T>
using scalar_real_t = typename ScalarRealType<T>::type;

template <typename Op>
typename Op::VectorType make_seed_vector(const Op& op) {
  using VectorType = typename Op::VectorType;
  using ScalarType = typename Op::ScalarType;

  VectorType seed(op.dimension());
  seed.fill(static_cast<ScalarType>(1));
  return seed;
}

template <typename Vector>
struct LinearOperatorTraits {
  using ScalarType = typename Vector::elem_type;
};

template <typename Vector>
struct LinearOperator {
  using VectorType = Vector;
  using ScalarType = typename LinearOperatorTraits<Vector>::ScalarType;

  virtual ~LinearOperator() = default;

  virtual VectorType apply(const VectorType& v) const = 0;
  virtual size_t dimension() const = 0;
};

template <typename Op>
struct Negated final : LinearOperator<typename Op::VectorType> {
  using VectorType = typename Op::VectorType;
  using ScalarType = typename Op::ScalarType;

  explicit Negated(Op op) : op_(std::move(op)) {}

  VectorType apply(const VectorType& v) const override { return -op_.apply(v); }
  size_t dimension() const override { return op_.dimension(); }

 private:
  Op op_;
};

template <typename Op>
struct Scaled final : LinearOperator<typename Op::VectorType> {
  using VectorType = typename Op::VectorType;
  using ScalarType = typename Op::ScalarType;

  Scaled(Op op, ScalarType scale) : op_(std::move(op)), scale_(scale) {}

  VectorType apply(const VectorType& v) const override { return scale_ * op_.apply(v); }
  size_t dimension() const override { return op_.dimension(); }

 private:
  Op op_;
  ScalarType scale_{};
};

template <typename Op>
struct Shifted final : LinearOperator<typename Op::VectorType> {
  using VectorType = typename Op::VectorType;
  using ScalarType = typename Op::ScalarType;

  Shifted(Op op, ScalarType shift) : op_(std::move(op)), shift_(shift) {}

  VectorType apply(const VectorType& v) const override { return op_.apply(v) + shift_ * v; }
  size_t dimension() const override { return op_.dimension(); }

 private:
  Op op_;
  ScalarType shift_{};
};

template <typename Op>
struct NegShift final : LinearOperator<typename Op::VectorType> {
  using VectorType = typename Op::VectorType;
  using ScalarType = typename Op::ScalarType;

  NegShift(Op op, ScalarType shift) : op_(std::move(op)), shift_(shift) {}

  VectorType apply(const VectorType& v) const override { return -op_.apply(v) + shift_ * v; }
  size_t dimension() const override { return op_.dimension(); }

 private:
  Op op_;
  ScalarType shift_{};
};

template <typename LeftOp, typename RightOp>
struct Sum final : LinearOperator<typename LeftOp::VectorType> {
  using VectorType = typename LeftOp::VectorType;
  using ScalarType = typename LeftOp::ScalarType;

  static_assert(std::is_same_v<VectorType, typename RightOp::VectorType>,
                "Sum requires matching vector types.");
  static_assert(std::is_same_v<ScalarType, typename RightOp::ScalarType>,
                "Sum requires matching scalar types.");

  Sum(LeftOp lhs, RightOp rhs) : lhs_(std::move(lhs)), rhs_(std::move(rhs)) {
    assert(lhs_.dimension() == rhs_.dimension());
  }

  VectorType apply(const VectorType& v) const override { return lhs_.apply(v) + rhs_.apply(v); }
  size_t dimension() const override { return lhs_.dimension(); }

 private:
  LeftOp lhs_;
  RightOp rhs_;
};

template <typename LeftOp, typename RightOp>
struct Difference final : LinearOperator<typename LeftOp::VectorType> {
  using VectorType = typename LeftOp::VectorType;
  using ScalarType = typename LeftOp::ScalarType;

  static_assert(std::is_same_v<VectorType, typename RightOp::VectorType>,
                "Difference requires matching vector types.");
  static_assert(std::is_same_v<ScalarType, typename RightOp::ScalarType>,
                "Difference requires matching scalar types.");

  Difference(LeftOp lhs, RightOp rhs) : lhs_(std::move(lhs)), rhs_(std::move(rhs)) {
    assert(lhs_.dimension() == rhs_.dimension());
  }

  VectorType apply(const VectorType& v) const override { return lhs_.apply(v) - rhs_.apply(v); }
  size_t dimension() const override { return lhs_.dimension(); }

 private:
  LeftOp lhs_;
  RightOp rhs_;
};

template <typename LeftOp, typename RightOp>
struct Composed final : LinearOperator<typename LeftOp::VectorType> {
  using VectorType = typename LeftOp::VectorType;
  using ScalarType = typename LeftOp::ScalarType;

  static_assert(std::is_same_v<VectorType, typename RightOp::VectorType>,
                "Composed requires matching vector types.");
  static_assert(std::is_same_v<ScalarType, typename RightOp::ScalarType>,
                "Composed requires matching scalar types.");

  Composed(LeftOp lhs, RightOp rhs) : lhs_(std::move(lhs)), rhs_(std::move(rhs)) {
    assert(lhs_.dimension() == rhs_.dimension());
  }

  VectorType apply(const VectorType& v) const override { return lhs_.apply(rhs_.apply(v)); }
  size_t dimension() const override { return lhs_.dimension(); }

 private:
  LeftOp lhs_;
  RightOp rhs_;
};

template <typename Vector>
struct Identity final : LinearOperator<Vector> {
  using VectorType = Vector;
  using ScalarType = typename LinearOperator<Vector>::ScalarType;

  explicit Identity(size_t dimension) : dimension_(dimension) {}

  VectorType apply(const VectorType& v) const override { return v; }
  size_t dimension() const override { return dimension_; }

 private:
  size_t dimension_{};
};

template <typename Scalar>
struct ExpOptions {
  using RealType = scalar_real_t<Scalar>;

  size_t power_iterations = 30;
  size_t max_degree = 200;
  RealType tolerance = static_cast<RealType>(1e-8);
  RealType scale_target = static_cast<RealType>(20);
  size_t max_scale_steps = 16;
  RealType spectral_padding = static_cast<RealType>(0.1);
};

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

template <typename Op>
struct Exp final : LinearOperator<typename Op::VectorType> {
  using VectorType = typename Op::VectorType;
  using ScalarType = typename Op::ScalarType;
  using RealType = scalar_real_t<ScalarType>;
  using Options = ExpOptions<ScalarType>;

  Exp(Op op, Options options = {}) : op_(std::move(op)), options_(options) {
    bounds_ = estimate_bounds(op_, options_.power_iterations, options_.spectral_padding);
    const auto [alpha, beta] = bounds_;
    a_ = (beta - alpha) / static_cast<RealType>(2);
    c_ = (beta + alpha) / static_cast<RealType>(2);
    if (a_ <= std::numeric_limits<RealType>::epsilon()) {
      trivial_ = true;
      exp_c_ = static_cast<ScalarType>(std::exp(c_));
      return;
    }

    steps_ = 1;
    if (options_.scale_target > static_cast<RealType>(0) && a_ > options_.scale_target) {
      steps_ = static_cast<size_t>(std::ceil(a_ / options_.scale_target));
      if (options_.max_scale_steps > 0) {
        steps_ = std::min(steps_, options_.max_scale_steps);
      }
      steps_ = std::max<size_t>(1, steps_);
    }

    c_step_ = c_ / static_cast<RealType>(steps_);
    a_step_ = a_ / static_cast<RealType>(steps_);
    coeffs_ = compute_coeffs(a_step_);
    exp_c_step_ = static_cast<ScalarType>(std::exp(c_step_));
  }

  VectorType apply(const VectorType& v) const override {
    const auto v_norm = arma::norm(v);
    if (v_norm == static_cast<RealType>(0)) {
      return v;
    }

    if (trivial_) {
      return exp_c_ * v;
    }

    VectorType result = v;
    for (size_t step = 0; step < steps_; ++step) {
      result = chebyshev_apply(result, coeffs_, exp_c_step_, c_, a_);
    }
    return result;
  }

  size_t dimension() const override { return op_.dimension(); }

 private:
  std::vector<RealType> compute_coeffs(RealType a) const {
    std::vector<RealType> coeffs;
    coeffs.reserve(options_.max_degree + 1);
    const size_t cutoff = static_cast<size_t>(std::max(static_cast<RealType>(1), std::ceil(a)));
    for (size_t k = 0; k <= options_.max_degree; ++k) {
      const RealType ik = bessel_i(k, a, options_.tolerance);
      const RealType coeff = (k == 0) ? ik : static_cast<RealType>(2) * ik;
      coeffs.push_back(coeff);
      if (coeff < options_.tolerance && k > cutoff) {
        break;
      }
    }
    return coeffs;
  }

  VectorType apply_B(const VectorType& v, RealType c, RealType a) const {
    return (op_.apply(v) - static_cast<ScalarType>(c) * v) / static_cast<ScalarType>(a);
  }

  VectorType chebyshev_apply(const VectorType& v, const std::vector<RealType>& coeffs,
                             ScalarType exp_c, RealType c, RealType a) const {
    if (coeffs.size() == 1) {
      return exp_c * static_cast<ScalarType>(coeffs.front()) * v;
    }

    VectorType w0 = v;
    VectorType w1 = apply_B(v, c, a);
    VectorType y =
        static_cast<ScalarType>(coeffs[0]) * w0 + static_cast<ScalarType>(coeffs[1]) * w1;
    for (size_t k = 1; k + 1 < coeffs.size(); ++k) {
      VectorType w2 = static_cast<ScalarType>(2) * apply_B(w1, c, a) - w0;
      y += static_cast<ScalarType>(coeffs[k + 1]) * w2;
      w0 = std::move(w1);
      w1 = std::move(w2);
    }
    return exp_c * y;
  }

  Op op_;
  Options options_;
  Bounds<ScalarType> bounds_{};
  size_t steps_{1};
  RealType a_{0};
  RealType c_{0};
  RealType c_step_{0};
  RealType a_step_{0};
  ScalarType exp_c_{0};
  ScalarType exp_c_step_{0};
  bool trivial_{false};
  std::vector<RealType> coeffs_{};
};

template <typename T, typename = void>
struct is_linear_operator : std::false_type {};

template <typename T>
struct is_linear_operator<T, std::void_t<typename T::VectorType, typename T::ScalarType>>
    : std::bool_constant<std::is_base_of_v<LinearOperator<typename T::VectorType>, T>> {};

template <typename T>
inline constexpr bool is_linear_operator_v = is_linear_operator<T>::value;

template <typename Op, std::enable_if_t<is_linear_operator_v<Op>, int> = 0>
Negated<Op> operator-(const Op& op) {
  return Negated<Op>(op);
}

template <typename LeftOp, typename RightOp,
          std::enable_if_t<is_linear_operator_v<LeftOp> && is_linear_operator_v<RightOp>, int> = 0>
Sum<LeftOp, RightOp> operator+(const LeftOp& lhs, const RightOp& rhs) {
  return Sum<LeftOp, RightOp>(lhs, rhs);
}

template <typename LeftOp, typename RightOp,
          std::enable_if_t<is_linear_operator_v<LeftOp> && is_linear_operator_v<RightOp>, int> = 0>
Difference<LeftOp, RightOp> operator-(const LeftOp& lhs, const RightOp& rhs) {
  return Difference<LeftOp, RightOp>(lhs, rhs);
}

template <typename LeftOp, typename RightOp,
          std::enable_if_t<is_linear_operator_v<LeftOp> && is_linear_operator_v<RightOp>, int> = 0>
Composed<LeftOp, RightOp> operator*(const LeftOp& lhs, const RightOp& rhs) {
  return Composed<LeftOp, RightOp>(lhs, rhs);
}

template <
    typename Op, typename Scalar,
    std::enable_if_t<is_linear_operator_v<Op> && !is_linear_operator_v<std::decay_t<Scalar>> &&
                         std::is_convertible_v<Scalar, typename Op::ScalarType>,
                     int> = 0>
Scaled<Op> operator*(const Op& op, Scalar scale) {
  return Scaled<Op>(op, static_cast<typename Op::ScalarType>(scale));
}

template <
    typename Scalar, typename Op,
    std::enable_if_t<is_linear_operator_v<Op> && !is_linear_operator_v<std::decay_t<Scalar>> &&
                         std::is_convertible_v<Scalar, typename Op::ScalarType>,
                     int> = 0>
Scaled<Op> operator*(Scalar scale, const Op& op) {
  return Scaled<Op>(op, static_cast<typename Op::ScalarType>(scale));
}
