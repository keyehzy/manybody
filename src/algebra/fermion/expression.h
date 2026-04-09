#pragma once

#include "algebra/canonicalize.h"
#include "algebra/expression_base.h"
#include "algebra/fermion/term.h"

struct FermionExpression : ExpressionBase<FermionExpression, FermionMonomial> {
  using Base = ExpressionBase<FermionExpression, FermionMonomial>;
  using Base::Base;
};

FermionExpression hopping(const FermionExpression::complex_type& coeff, size_t from, size_t to,
                          FermionOperator::Spin spin) noexcept;
FermionExpression hopping(size_t from, size_t to, FermionOperator::Spin spin) noexcept;

inline FermionExpression canonicalize(const FermionMonomial::complex_type& c,
                                      const FermionMonomial::container_type& ops) {
  return canonicalize_generic<FermionExpression>(c, ops);
}

inline FermionExpression canonicalize(const FermionMonomial& term) {
  return canonicalize_generic<FermionExpression>(term.c, term.operators);
}

inline FermionExpression canonicalize(const FermionExpression& expr) {
  return canonicalize_generic<FermionExpression>(expr);
}
