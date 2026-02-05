#include "algebra/expression.h"

#include <algorithm>
#include <sstream>
#include <utility>

FermionExpression FermionExpression::adjoint() const {
  FermionExpression result;
  for (const auto& [ops, coeff] : this->data) {
    FermionMonomial term(coeff, ops);
    FermionMonomial adj = ::adjoint(term);
    result.add_to_map(std::move(adj.operators), adj.c);
  }
  return result;
}

void FermionExpression::format_to(std::ostringstream& oss) const {
  this->format_sorted(
      oss, [](std::ostringstream& os, const container_type& ops, const complex_type& coeff) {
        FermionMonomial term(coeff, ops);
        ::to_string(os, term);
      });
}

std::string FermionExpression::to_string() const {
  return Base::to_string(
      [](std::ostringstream& os, const container_type& ops, const complex_type& coeff) {
        FermionMonomial term(coeff, ops);
        ::to_string(os, term);
      });
}

