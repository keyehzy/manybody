#pragma once

#include <sstream>
#include <string>

#include "algebra/boson/term.h"
#include "algebra/expression_base.h"
#include "robin_hood.h"

struct BosonExpression : ExpressionBase<BosonExpression, BosonMonomial> {
  using Base = ExpressionBase<BosonExpression, BosonMonomial>;
  using Base::Base;

  void format_to(std::ostringstream& oss) const;
  std::string to_string() const;
};

BosonExpression adjoint(const BosonExpression& expr);

BosonExpression canonicalize(const BosonMonomial::complex_type& c,
                             const BosonMonomial::container_type& ops);
BosonExpression canonicalize(const BosonMonomial& term);
BosonExpression canonicalize(const BosonExpression& expr);
