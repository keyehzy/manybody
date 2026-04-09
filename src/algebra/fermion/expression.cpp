#include "algebra/fermion/expression.h"

#include <algorithm>
#include <complex>
#include <sstream>
#include <utility>

#include "algebra/canonicalize.h"
#include "utils/tolerances.h"

FermionExpression adjoint(const FermionExpression& expr) {
  FermionExpression result;
  for (const auto& [ops, coeff] : expr.terms()) {
    FermionMonomial term(coeff, ops);
    FermionMonomial adj = ::adjoint(term);
    result.add_to_map(std::move(adj.operators), adj.c);
  }
  return result;
}

FermionExpression hopping(const FermionExpression::complex_type& coeff, size_t from, size_t to,
                          Operator::Spin spin) noexcept {
  FermionExpression result = FermionExpression(
      FermionMonomial(coeff, {Operator::creation(spin, from), Operator::annihilation(spin, to)}));
  result += FermionExpression(FermionMonomial(
      std::conj(coeff), {Operator::creation(spin, to), Operator::annihilation(spin, from)}));
  return result;
}

FermionExpression hopping(size_t from, size_t to, Operator::Spin spin) noexcept {
  return hopping(1.0, from, to, spin);
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

Expression canonicalize(const FermionMonomial::complex_type& c,
                        const FermionMonomial::container_type& ops) {
  return canonicalize_generic<Expression>(c, ops);
}

Expression canonicalize(const FermionMonomial& term) {
  return canonicalize_generic<Expression>(term.c, term.operators);
}

Expression canonicalize(const Expression& expr) {
  return canonicalize_generic<Expression>(expr);
}
