#pragma once

#include "algebra/canonicalize.h"
#include "algebra/expression_base.h"
#include "algebra/fermion/term.h"

struct FermionExpression : ExpressionBase<FermionExpression, FermionMonomial> {
  using Base = ExpressionBase<FermionExpression, FermionMonomial>;
  using Base::Base;
};

FermionExpression hopping(const FermionExpression::complex_type& coeff, size_t from, size_t to,
                          Operator::Spin spin) noexcept;
FermionExpression hopping(size_t from, size_t to, Operator::Spin spin) noexcept;

// Backwards compatibility alias
using Expression = FermionExpression;

inline Expression canonicalize(const FermionMonomial::complex_type& c,
                               const FermionMonomial::container_type& ops) {
  return canonicalize_generic<Expression>(c, ops);
}

inline Expression canonicalize(const FermionMonomial& term) {
  return canonicalize_generic<Expression>(term.c, term.operators);
}

inline Expression canonicalize(const Expression& expr) {
  return canonicalize_generic<Expression>(expr);
}
