#pragma once

#include <cstddef>
#include <initializer_list>
#include <utility>

#include "utils/static_vector.h"

template <typename Derived, typename OperatorType, size_t MaxOps, typename SizeType,
          typename Scalar>
struct MonomialBase {
  using operator_type = OperatorType;
  using scalar_type = Scalar;
  using container_type = static_vector<OperatorType, MaxOps, SizeType>;
  using complex_type = Scalar;

  scalar_type c{1.0};
  container_type operators{};

  constexpr MonomialBase() noexcept = default;
  constexpr ~MonomialBase() noexcept = default;

  constexpr MonomialBase(const MonomialBase&) noexcept = default;
  constexpr MonomialBase& operator=(const MonomialBase&) noexcept = default;
  constexpr MonomialBase(MonomialBase&&) noexcept = default;
  constexpr MonomialBase& operator=(MonomialBase&&) noexcept = default;

  explicit constexpr MonomialBase(operator_type op) noexcept : operators({op}) {}
  explicit constexpr MonomialBase(scalar_type x) noexcept : c(x) {}
  explicit constexpr MonomialBase(const container_type& ops) noexcept : operators(ops) {}
  explicit constexpr MonomialBase(container_type&& ops) noexcept : operators(std::move(ops)) {}
  explicit constexpr MonomialBase(scalar_type x, const container_type& ops) noexcept
      : c(x), operators(ops) {}
  explicit constexpr MonomialBase(scalar_type x, container_type&& ops) noexcept
      : c(x), operators(std::move(ops)) {}
  explicit constexpr MonomialBase(std::initializer_list<operator_type> init) noexcept
      : operators(init) {}
  explicit constexpr MonomialBase(scalar_type x, std::initializer_list<operator_type> init) noexcept
      : c(x), operators(init) {}

  constexpr size_t size() const noexcept { return operators.size(); }

  constexpr Derived& operator*=(const Derived& value) noexcept {
    c *= value.c;
    operators.append_range(value.operators.begin(), value.operators.end());
    return static_cast<Derived&>(*this);
  }

  constexpr Derived& operator*=(operator_type value) noexcept {
    operators.push_back(value);
    return static_cast<Derived&>(*this);
  }

  constexpr Derived& operator*=(scalar_type value) noexcept {
    c *= value;
    return static_cast<Derived&>(*this);
  }

  constexpr Derived& operator/=(scalar_type value) noexcept {
    c /= value;
    return static_cast<Derived&>(*this);
  }

  constexpr bool operator==(const Derived& other) const noexcept {
    return c == other.c && operators == other.operators;
  }

  friend constexpr Derived operator*(Derived a, const Derived& b) noexcept {
    a *= b;
    return a;
  }

  friend constexpr Derived operator*(Derived a, operator_type b) noexcept {
    a *= b;
    return a;
  }

  friend constexpr Derived operator*(Derived a, scalar_type b) noexcept {
    a *= b;
    return a;
  }

  friend constexpr Derived operator/(Derived a, scalar_type b) noexcept {
    a /= b;
    return a;
  }

  friend constexpr Derived operator*(scalar_type a, Derived b) noexcept {
    b *= a;
    return b;
  }

  friend constexpr Derived operator*(operator_type a, const Derived& b) noexcept {
    Derived result(a);
    result *= b;
    return result;
  }
};

template <typename OperatorType, size_t MaxOps, typename SizeType, typename Scalar>
struct MonomialImpl : MonomialBase<MonomialImpl<OperatorType, MaxOps, SizeType, Scalar>,
                                   OperatorType, MaxOps, SizeType, Scalar> {
  using Base = MonomialBase<MonomialImpl, OperatorType, MaxOps, SizeType, Scalar>;
  using Base::Base;
};
