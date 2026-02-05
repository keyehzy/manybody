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

  void format_to(std::ostringstream& oss) const;
  std::string to_string() const;
};

FermionExpression adjoint(const FermionExpression& expr);

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

Expression normal_order(const FermionMonomial::complex_type& c,
                        const FermionMonomial::container_type& ops);
Expression normal_order(const FermionMonomial& term);
Expression normal_order(const Expression& expr);

Expression canonicalize(const FermionMonomial::complex_type& c,
                        const FermionMonomial::container_type& ops);
Expression canonicalize(const FermionMonomial& term);
Expression canonicalize(const Expression& expr);
