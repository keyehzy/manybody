#pragma once

#include <algorithm>
#include <sstream>
#include <string>

#include "algebra/expression_base.h"
#include "algebra/fermion/term.h"
#include "robin_hood.h"

struct FermionExpression : ExpressionBase<FermionExpression, FermionMonomial> {
  using Base = ExpressionBase<FermionExpression, FermionMonomial>;
  using Base::Base;

  void format_to(std::ostringstream& oss) const;
  std::string to_string() const;
};

FermionExpression adjoint(const FermionExpression& expr);

FermionExpression hopping(const FermionExpression::complex_type& coeff, size_t from, size_t to,
                          Operator::Spin spin) noexcept;
FermionExpression hopping(size_t from, size_t to, Operator::Spin spin) noexcept;

// Backwards compatibility alias
using Expression = FermionExpression;

Expression canonicalize(const FermionMonomial::complex_type& c,
                        const FermionMonomial::container_type& ops);
Expression canonicalize(const FermionMonomial& term);
Expression canonicalize(const Expression& expr);
