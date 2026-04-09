#pragma once

#include "algebra/boson/term.h"
#include "algebra/canonicalize.h"
#include "algebra/expression_base.h"

struct BosonExpression : ExpressionBase<BosonExpression, BosonMonomial> {
  using Base = ExpressionBase<BosonExpression, BosonMonomial>;
  using Base::Base;
};

BosonExpression hopping(const BosonExpression::complex_type& coeff, size_t from, size_t to,
                        BosonOperator::Spin spin) noexcept;
BosonExpression hopping(size_t from, size_t to, BosonOperator::Spin spin) noexcept;

inline BosonExpression canonicalize(const BosonMonomial::complex_type& c,
                                    const BosonMonomial::container_type& ops) {
  return canonicalize_generic<BosonExpression>(c, ops);
}

inline BosonExpression canonicalize(const BosonMonomial& term) {
  return canonicalize_generic<BosonExpression>(term.c, term.operators);
}

inline BosonExpression canonicalize(const BosonExpression& expr) {
  return canonicalize_generic<BosonExpression>(expr);
}
