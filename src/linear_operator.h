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

struct MatrixOperator final : LinearOperator<arma::vec> {
  using VectorType = arma::vec;
  using ScalarType = double;

  explicit MatrixOperator(arma::mat matrix_in) : matrix(std::move(matrix_in)) {}

  VectorType apply(const VectorType& v) const override { return matrix * v; }
  size_t dimension() const override { return static_cast<size_t>(matrix.n_rows); }

  arma::mat matrix;
};

struct DiagonalOperator final : LinearOperator<arma::vec> {
  using VectorType = arma::vec;
  using ScalarType = double;

  explicit DiagonalOperator(arma::vec diag_in) : diag(std::move(diag_in)) {}

  VectorType apply(const VectorType& v) const override { return diag % v; }
  size_t dimension() const override { return static_cast<size_t>(diag.n_elem); }

  arma::vec diag;
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
