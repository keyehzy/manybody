#pragma once

#include <algorithm>
#include <sstream>
#include <string>

#include "algebra/expression_base.h"
#include "algebra/term.h"
#include "robin_hood.h"

struct FermionExpression : ExpressionBase<FermionExpression, FermionMonomial> {
  using Base = ExpressionBase<FermionExpression, FermionMonomial>;
  using Base::operator*=;
  using Base::operator+=;
  using Base::operator-=;

  // Helper to add to map with size check
  void add_to_map(const container_type& ops, const complex_type& coeff) {
    if (ops.size() > FermionMonomial::container_type::max_size()) {
      return;
    }
    this->add(ops, coeff);
  }

  void add_to_map(container_type&& ops, const complex_type& coeff) {
    if (ops.size() > FermionMonomial::container_type::max_size()) {
      return;
    }
    this->add(std::move(ops), coeff);
  }

  FermionExpression() = default;

  explicit FermionExpression(complex_type c) {
    if (!is_zero(c)) {
      data.emplace(container_type{}, c);
    }
  }

  explicit FermionExpression(const FermionMonomial& term) { add_to_map(term.operators, term.c); }

  explicit FermionExpression(FermionMonomial&& term) {
    add_to_map(std::move(term.operators), term.c);
  }

  explicit FermionExpression(const container_type& container) {
    data.emplace(container, complex_type{1.0, 0.0});
  }

  explicit FermionExpression(container_type&& container) {
    data.emplace(std::move(container), complex_type{1.0, 0.0});
  }

  template <typename Container>
  FermionExpression(complex_type c, Container&& ops) {
    data.emplace(std::forward<Container>(ops), c);
  }

  explicit FermionExpression(Operator op) {
    container_type ops{op};
    data.emplace(std::move(ops), complex_type{1.0, 0.0});
  }

  explicit FermionExpression(std::initializer_list<FermionMonomial> lst) {
    for (const auto& term : lst) {
      add_to_map(term.operators, term.c);
    }
  }

  FermionExpression adjoint() const;

  FermionExpression& truncate_by_size(size_t max_size);
  FermionExpression& filter_by_size(size_t size);

  FermionExpression& operator*=(const FermionExpression& value);
  FermionExpression& operator*=(const FermionMonomial& value);

  void format_to(std::ostringstream& oss) const;
  std::string to_string() const;
};

inline FermionExpression operator+(FermionExpression lhs, const FermionExpression& rhs) {
  lhs += rhs;
  return lhs;
}

inline FermionExpression operator-(FermionExpression lhs, const FermionExpression& rhs) {
  lhs -= rhs;
  return lhs;
}

inline FermionExpression operator*(FermionExpression lhs, const FermionExpression& rhs) {
  lhs *= rhs;
  return lhs;
}

inline FermionExpression operator*(FermionExpression lhs,
                                   const FermionExpression::complex_type& rhs) {
  lhs *= rhs;
  return lhs;
}

inline FermionExpression operator*(const FermionExpression::complex_type& lhs,
                                   FermionExpression rhs) {
  rhs *= lhs;
  return rhs;
}

inline FermionExpression hopping(const FermionExpression::complex_type& coeff, size_t from,
                                 size_t to, Operator::Spin spin) noexcept {
  FermionExpression result = FermionExpression(
      FermionMonomial(coeff, {Operator::creation(spin, from), Operator::annihilation(spin, to)}));
  result += FermionExpression(FermionMonomial(
      std::conj(coeff), {Operator::creation(spin, to), Operator::annihilation(spin, from)}));
  return result;
}

inline FermionExpression hopping(size_t from, size_t to, Operator::Spin spin) noexcept {
  return hopping(1.0, from, to, spin);
}

// Backwards compatibility alias
using Expression = FermionExpression;
