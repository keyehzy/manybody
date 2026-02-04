#pragma once

#include <cstddef>
#include <initializer_list>
#include <utility>

#include "utils/static_vector.h"

template <typename OperatorType, size_t MaxOps, typename SizeType, typename Scalar>
struct Monomial {
  using operator_type = OperatorType;
  using scalar_type = Scalar;
  using container_type = static_vector<OperatorType, MaxOps, SizeType>;

  scalar_type c{1.0};
  container_type operators{};

  constexpr Monomial() noexcept = default;
  constexpr ~Monomial() noexcept = default;

  constexpr Monomial(const Monomial&) noexcept = default;
  constexpr Monomial& operator=(const Monomial&) noexcept = default;
  constexpr Monomial(Monomial&&) noexcept = default;
  constexpr Monomial& operator=(Monomial&&) noexcept = default;

  explicit constexpr Monomial(operator_type op) noexcept : operators({op}) {}
  explicit constexpr Monomial(scalar_type x) noexcept : c(x) {}
  explicit constexpr Monomial(const container_type& ops) noexcept : operators(ops) {}
  explicit constexpr Monomial(container_type&& ops) noexcept : operators(std::move(ops)) {}
  explicit constexpr Monomial(scalar_type x, const container_type& ops) noexcept
      : c(x), operators(ops) {}
  explicit constexpr Monomial(scalar_type x, container_type&& ops) noexcept
      : c(x), operators(std::move(ops)) {}
  explicit constexpr Monomial(std::initializer_list<operator_type> init) noexcept
      : operators(init) {}
  explicit constexpr Monomial(scalar_type x, std::initializer_list<operator_type> init) noexcept
      : c(x), operators(init) {}

  constexpr size_t size() const noexcept { return operators.size(); }

  constexpr Monomial& operator*=(const Monomial& value) noexcept {
    c *= value.c;
    operators.append_range(value.operators.begin(), value.operators.end());
    return *this;
  }

  constexpr Monomial& operator*=(operator_type value) noexcept {
    operators.push_back(value);
    return *this;
  }

  constexpr Monomial& operator*=(scalar_type value) noexcept {
    c *= value;
    return *this;
  }

  constexpr Monomial& operator/=(scalar_type value) noexcept {
    c /= value;
    return *this;
  }
};
