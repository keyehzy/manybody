#include "algebra/expression.h"

#include <algorithm>
#include <sstream>
#include <utility>

#include "utils/tolerances.h"

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

using complex_type = FermionMonomial::complex_type;
using container_type = FermionMonomial::container_type;

constexpr auto tolerance = tolerances::tolerance<Expression::complex_type::value_type>();

namespace {
Expression normal_order_recursive(container_type ops);
Expression handle_non_commuting(container_type ops, size_t index);

Expression normal_order_recursive(container_type ops) {
  if (ops.size() < 2) {
    return Expression(ops);
  }

  if (has_consecutive_elements(ops)) {
    return {};
  }

  complex_type phase{1.0, 0.0};

  for (size_t i = 1; i < ops.size(); ++i) {
    size_t j = i;
    while (j > 0 && ops[j] < ops[j - 1]) {
      if (ops[j].commutes(ops[j - 1])) {
        std::swap(ops[j], ops[j - 1]);
        phase = -phase;
        --j;
      } else {
        Expression result = handle_non_commuting(std::move(ops), j - 1);
        result *= phase;
        return result;
      }
    }
  }

  if (has_consecutive_elements(ops)) {
    return {};
  }

  return Expression(phase, std::move(ops));
}

Expression handle_non_commuting(container_type ops, size_t index) {
  container_type contracted;
  contracted.append_range(ops.begin(), ops.begin() + index);
  contracted.append_range(ops.begin() + index + 2, ops.end());
  std::swap(ops[index], ops[index + 1]);
  Expression lhs = normal_order_recursive(std::move(contracted));
  lhs -= normal_order_recursive(std::move(ops));
  return lhs;
}
}  // namespace

Expression normal_order(const complex_type& c, const container_type& ops) {
  if (std::norm(c) < tolerance * tolerance) {
    return {};
  }
  Expression result = normal_order_recursive(ops);
  result *= c;
  return result;
}

Expression normal_order(const FermionMonomial& term) {
  return normal_order(term.c, term.operators);
}

Expression normal_order(const Expression& expr) {
  Expression result;
  for (const auto& [ops, c] : expr.terms()) {
    result += normal_order(c, ops);
  }
  return result;
}

// For backwards compatibility
Expression canonicalize(const FermionMonomial::complex_type& c,
                        const FermionMonomial::container_type& ops) {
  return normal_order(c, ops);
}

Expression canonicalize(const FermionMonomial& term) { return normal_order(term); }

Expression canonicalize(const Expression& expr) { return normal_order(expr); }
