#pragma once

#include "algebra/canonicalize.h"
#include "algebra/expression_base.h"
#include "algebra/hardcore_boson/term.h"

struct HardcoreBosonExpression : ExpressionBase<HardcoreBosonExpression, HardcoreBosonMonomial> {
  using Base = ExpressionBase<HardcoreBosonExpression, HardcoreBosonMonomial>;
  using Base::Base;
};

HardcoreBosonExpression hopping(const HardcoreBosonExpression::complex_type& coeff, size_t from,
                                size_t to, HardcoreBosonOperator::Spin spin) noexcept;
HardcoreBosonExpression hopping(size_t from, size_t to,
                                HardcoreBosonOperator::Spin spin) noexcept;

inline HardcoreBosonExpression canonicalize(const HardcoreBosonMonomial::complex_type& c,
                                            const HardcoreBosonMonomial::container_type& ops) {
  return canonicalize_generic<HardcoreBosonExpression>(c, ops);
}

inline HardcoreBosonExpression canonicalize(const HardcoreBosonMonomial& term) {
  return canonicalize_generic<HardcoreBosonExpression>(term.c, term.operators);
}

inline HardcoreBosonExpression canonicalize(const HardcoreBosonExpression& expr) {
  return canonicalize_generic<HardcoreBosonExpression>(expr);
}
