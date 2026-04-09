#include "algebra/boson/expression.h"

#include <complex>
#include <sstream>
#include <utility>

#include "algebra/canonicalize.h"

BosonExpression adjoint(const BosonExpression& expr) {
  BosonExpression result;
  for (const auto& [ops, coeff] : expr.terms()) {
    BosonMonomial term(coeff, ops);
    BosonMonomial adj = ::adjoint(term);
    result.add_to_map(std::move(adj.operators), adj.c);
  }
  return result;
}

void BosonExpression::format_to(std::ostringstream& oss) const {
  this->format_sorted(
      oss, [](std::ostringstream& os, const container_type& ops, const complex_type& coeff) {
        BosonMonomial term(coeff, ops);
        ::to_string(os, term);
      });
}

std::string BosonExpression::to_string() const {
  return Base::to_string(
      [](std::ostringstream& os, const container_type& ops, const complex_type& coeff) {
        BosonMonomial term(coeff, ops);
        ::to_string(os, term);
      });
}

BosonExpression canonicalize(const BosonMonomial::complex_type& c,
                             const BosonMonomial::container_type& ops) {
  return canonicalize_generic<BosonExpression>(c, ops);
}

BosonExpression canonicalize(const BosonMonomial& term) {
  return canonicalize_generic<BosonExpression>(term.c, term.operators);
}

BosonExpression canonicalize(const BosonExpression& expr) {
  return canonicalize_generic<BosonExpression>(expr);
}
