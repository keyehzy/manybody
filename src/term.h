#pragma once

#include <complex>
#include <cstddef>
#include <cstdint>

#include "operator.h"
#include "static_vector.h"

constexpr size_t term_size = 16;

struct Term {
  using complex_type = std::complex<float>;

  static constexpr size_t static_vector_size =
      (term_size - sizeof(complex_type)) / sizeof(Operator) - 1;

  using container_type = static_vector<Operator, static_vector_size, uint8_t>;

  complex_type c{1.0f, 0.0f};
  container_type operators{};

  constexpr Term() noexcept = default;
  constexpr ~Term() noexcept = default;

  constexpr Term(const Term& other) noexcept = default;
  constexpr Term& operator=(const Term& other) noexcept = default;
  constexpr Term(Term&& other) noexcept = default;
  constexpr Term& operator=(Term&& other) noexcept = default;

  explicit constexpr Term(Operator x) noexcept : operators({x}) {}
  explicit constexpr Term(complex_type x) noexcept : c(x) {}
  explicit constexpr Term(const container_type& ops) noexcept
      : operators(ops) {}
  explicit constexpr Term(container_type&& ops) noexcept
      : operators(std::move(ops)) {}
  explicit constexpr Term(complex_type x, const container_type& ops) noexcept
      : c(x), operators(ops) {}
  explicit constexpr Term(complex_type x, container_type&& ops) noexcept
      : c(x), operators(std::move(ops)) {}
  explicit constexpr Term(std::initializer_list<Operator> init) noexcept
      : operators(init) {}
  explicit constexpr Term(complex_type x,
                          std::initializer_list<Operator> init) noexcept
      : c(x), operators(init) {}

  constexpr size_t size() const noexcept { return operators.size(); }

  std::string to_string() const;

  constexpr bool operator==(const Term& other) const noexcept {
    return c == other.c && operators == other.operators;
  }

  constexpr Term adjoint() const noexcept {
    Term result(std::conj(c));
    for (auto it = operators.rbegin(); it != operators.rend(); ++it) {
      result.operators.push_back(it->adjoint());
    }
    return result;
  }

  constexpr Term& operator*=(const Term& value) noexcept {
    c *= value.c;
    operators.append_range(value.operators.begin(), value.operators.end());
    return *this;
  }

  constexpr Term& operator*=(Operator value) noexcept {
    operators.push_back(value);
    return *this;
  }

  constexpr Term& operator*=(complex_type value) noexcept {
    c *= value;
    return *this;
  }

  constexpr Term& operator/=(complex_type value) noexcept {
    c /= value;
    return *this;
  }
};

inline constexpr Term operator*(const Term& a, const Term& b) noexcept {
  Term result(a);
  result *= b;
  return result;
}

inline constexpr Term operator*(const Term& a, Operator b) noexcept {
  Term result(a);
  result *= b;
  return result;
}

inline constexpr Term operator*(const Term& a, Term::complex_type b) noexcept {
  Term result(a);
  result *= b;
  return result;
}

inline constexpr Term operator/(const Term& a, Term::complex_type b) noexcept {
  Term result(a);
  result /= b;
  return result;
}

inline constexpr Term operator*(Term::complex_type a, const Term& b) noexcept {
  Term result(a);
  result *= b;
  return result;
}

inline constexpr Term operator*(Operator a, const Term& b) noexcept {
  Term result(a);
  result *= b;
  return result;
}
static_assert(sizeof(Term) == term_size);
