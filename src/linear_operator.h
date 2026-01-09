#pragma once

#include <cassert>
#include <cstddef>
#include <type_traits>

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

  explicit Negated(const Op& op) : op_(op) {}

  VectorType apply(const VectorType& v) const override { return -op_.apply(v); }
  size_t dimension() const override { return op_.dimension(); }

 private:
  const Op& op_;
};

template <typename Op>
struct Scaled final : LinearOperator<typename Op::VectorType> {
  using VectorType = typename Op::VectorType;
  using ScalarType = typename Op::ScalarType;

  Scaled(const Op& op, ScalarType scale) : op_(op), scale_(scale) {}

  VectorType apply(const VectorType& v) const override { return scale_ * op_.apply(v); }
  size_t dimension() const override { return op_.dimension(); }

 private:
  const Op& op_;
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

  Sum(const LeftOp& lhs, const RightOp& rhs) : lhs_(lhs), rhs_(rhs) {
    assert(lhs_.dimension() == rhs_.dimension());
  }

  VectorType apply(const VectorType& v) const override { return lhs_.apply(v) + rhs_.apply(v); }
  size_t dimension() const override { return lhs_.dimension(); }

 private:
  const LeftOp& lhs_;
  const RightOp& rhs_;
};

template <typename LeftOp, typename RightOp>
struct Difference final : LinearOperator<typename LeftOp::VectorType> {
  using VectorType = typename LeftOp::VectorType;
  using ScalarType = typename LeftOp::ScalarType;

  static_assert(std::is_same_v<VectorType, typename RightOp::VectorType>,
                "Difference requires matching vector types.");
  static_assert(std::is_same_v<ScalarType, typename RightOp::ScalarType>,
                "Difference requires matching scalar types.");

  Difference(const LeftOp& lhs, const RightOp& rhs) : lhs_(lhs), rhs_(rhs) {
    assert(lhs_.dimension() == rhs_.dimension());
  }

  VectorType apply(const VectorType& v) const override { return lhs_.apply(v) - rhs_.apply(v); }
  size_t dimension() const override { return lhs_.dimension(); }

 private:
  const LeftOp& lhs_;
  const RightOp& rhs_;
};

template <typename LeftOp, typename RightOp>
struct Composed final : LinearOperator<typename LeftOp::VectorType> {
  using VectorType = typename LeftOp::VectorType;
  using ScalarType = typename LeftOp::ScalarType;

  static_assert(std::is_same_v<VectorType, typename RightOp::VectorType>,
                "Composed requires matching vector types.");
  static_assert(std::is_same_v<ScalarType, typename RightOp::ScalarType>,
                "Composed requires matching scalar types.");

  Composed(const LeftOp& lhs, const RightOp& rhs) : lhs_(lhs), rhs_(rhs) {
    assert(lhs_.dimension() == rhs_.dimension());
  }

  VectorType apply(const VectorType& v) const override { return lhs_.apply(rhs_.apply(v)); }
  size_t dimension() const override { return lhs_.dimension(); }

 private:
  const LeftOp& lhs_;
  const RightOp& rhs_;
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
