#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <complex>
#include <cstddef>
#include <limits>
#include <type_traits>
#include <utility>
#include <vector>

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

template <typename T>
constexpr scalar_real_t<T> real_part(const T& value) {
  if constexpr (std::is_same_v<T, std::complex<scalar_real_t<T>>>) {
    return std::real(value);
  } else {
    return value;
  }
}

template <typename Vector>
auto vector_dot(const Vector& lhs, const Vector& rhs) {
  return dot(lhs, rhs);
}

template <typename Vector>
scalar_real_t<typename Vector::elem_type> vector_norm(const Vector& v) {
  auto value = vector_dot(v, v);
  return std::sqrt(real_part(value));
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

template <typename Op>
struct Exp final : LinearOperator<typename Op::VectorType> {
  using VectorType = typename Op::VectorType;
  using ScalarType = typename Op::ScalarType;
  using RealType = scalar_real_t<ScalarType>;
  using Options = ExpOptions<ScalarType>;

  Exp(Op op, Options options = {}) : op_(std::move(op)), options_(options) {}

  VectorType apply(const VectorType& v) const override {
    const auto v_norm = vector_norm(v);
    if (v_norm == static_cast<RealType>(0)) {
      return v;
    }

    const auto [alpha, beta] = estimate_bounds(v);
    RealType a = (beta - alpha) / static_cast<RealType>(2);
    RealType c = (beta + alpha) / static_cast<RealType>(2);
    if (a <= std::numeric_limits<RealType>::epsilon()) {
      return static_cast<ScalarType>(std::exp(c)) * v;
    }

    size_t steps = 1;
    if (options_.scale_target > static_cast<RealType>(0) && a > options_.scale_target) {
      steps = static_cast<size_t>(std::ceil(a / options_.scale_target));
      if (options_.max_scale_steps > 0) {
        steps = std::min(steps, options_.max_scale_steps);
      }
      steps = std::max<size_t>(1, steps);
    }

    const RealType c_step = c / static_cast<RealType>(steps);
    const RealType a_step = a / static_cast<RealType>(steps);
    const auto coeffs = compute_coeffs(a_step);
    VectorType result = v;
    for (size_t step = 0; step < steps; ++step) {
      result = chebyshev_apply(result, coeffs, c_step, c, a);
    }
    return result;
  }

  size_t dimension() const override { return op_.dimension(); }

 private:
  struct Bounds {
    RealType alpha;
    RealType beta;
  };

  Bounds estimate_bounds(const VectorType& seed) const {
    const RealType dominant = std::abs(power_method(op_, seed, options_.power_iterations));
    if (dominant == static_cast<RealType>(0)) {
      return {static_cast<RealType>(0), static_cast<RealType>(0)};
    }

    const RealType shift = dominant * (static_cast<RealType>(1) + options_.spectral_padding) +
                           std::numeric_limits<RealType>::epsilon();

    struct Shifted final : LinearOperator<VectorType> {
      const Op& op;
      ScalarType shift;

      Shifted(const Op& op_in, ScalarType shift_in) : op(op_in), shift(shift_in) {}

      VectorType apply(const VectorType& v) const override { return op.apply(v) + shift * v; }
      size_t dimension() const override { return op.dimension(); }
    };

    struct NegShifted final : LinearOperator<VectorType> {
      const Op& op;
      ScalarType shift;

      NegShifted(const Op& op_in, ScalarType shift_in) : op(op_in), shift(shift_in) {}

      VectorType apply(const VectorType& v) const override { return -op.apply(v) + shift * v; }
      size_t dimension() const override { return op.dimension(); }
    };

    Shifted shifted{op_, static_cast<ScalarType>(shift)};
    NegShifted neg_shifted{op_, static_cast<ScalarType>(shift)};

    const RealType beta = power_method(shifted, seed, options_.power_iterations) - shift;
    const RealType alpha = shift - power_method(neg_shifted, seed, options_.power_iterations);
    if (alpha <= beta) {
      return {alpha, beta};
    }
    return {beta, alpha};
  }

  template <typename OperatorType>
  RealType power_method(const OperatorType& op, VectorType v, size_t iterations) const {
    if (iterations == 0) {
      return static_cast<RealType>(0);
    }

    RealType norm = vector_norm(v);
    if (norm == static_cast<RealType>(0)) {
      return static_cast<RealType>(0);
    }
    v /= static_cast<ScalarType>(norm);

    RealType eigenvalue = static_cast<RealType>(0);
    for (size_t i = 0; i < iterations; ++i) {
      VectorType w = op.apply(v);
      const auto denom = vector_dot(v, v);
      const auto numerator = vector_dot(v, w);
      const RealType denom_real = real_part(denom);
      if (denom_real == static_cast<RealType>(0)) {
        return static_cast<RealType>(0);
      }
      eigenvalue = real_part(numerator) / denom_real;

      norm = vector_norm(w);
      if (norm == static_cast<RealType>(0)) {
        return static_cast<RealType>(0);
      }
      v = w / static_cast<ScalarType>(norm);
    }
    return eigenvalue;
  }

  std::vector<RealType> compute_coeffs(RealType a) const {
    std::vector<RealType> coeffs;
    coeffs.reserve(options_.max_degree + 1);
    const size_t cutoff = static_cast<size_t>(std::max(static_cast<RealType>(1), std::ceil(a)));
    for (size_t k = 0; k <= options_.max_degree; ++k) {
      const RealType ik = bessel_i(k, a);
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
                             RealType c_step, RealType c, RealType a) const {
    const ScalarType exp_c = static_cast<ScalarType>(std::exp(c_step));
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

  RealType bessel_i(size_t order, RealType x) const {
    const RealType abs_x = std::abs(x);
    if (abs_x == static_cast<RealType>(0)) {
      return order == 0 ? static_cast<RealType>(1) : static_cast<RealType>(0);
    }

    const RealType half_x = abs_x / static_cast<RealType>(2);
    RealType term = static_cast<RealType>(1);
    for (size_t k = 1; k <= order; ++k) {
      term *= half_x / static_cast<RealType>(k);
    }
    RealType sum = term;
    const RealType eps = std::max(options_.tolerance * static_cast<RealType>(0.1),
                                  std::numeric_limits<RealType>::epsilon());
    for (size_t m = 1; m < 200; ++m) {
      term *= (half_x * half_x) / static_cast<RealType>(m * (m + order));
      sum += term;
      if (term < eps * sum) {
        break;
      }
    }
    return sum;
  }

  Op op_;
  Options options_;
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
