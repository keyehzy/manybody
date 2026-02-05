#pragma once

#include <algorithm>
#include <sstream>
#include <string>

#include "algebra/expression_base.h"
#include "algebra/term.h"
#include "robin_hood.h"

struct FermionExpression : ExpressionBase<FermionExpression, FermionMonomial> {
  using Base = ExpressionBase<FermionExpression, FermionMonomial>;
  using Base::Base;
  using Base::operator*=;
  using Base::operator+=;
  using Base::operator-=;

  FermionExpression() = default;

  FermionExpression adjoint() const;

  void format_to(std::ostringstream& oss) const;
  std::string to_string() const;
};

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
