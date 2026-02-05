#include "algebra/normal_order.h"

#include "algebra/expression.h"
#include "utils/tolerances.h"

using complex_type = FermionMonomial::complex_type;
using container_type = FermionMonomial::container_type;

constexpr auto tolerance = tolerances::tolerance<Expression::complex_type::value_type>();

namespace {
Expression normal_order_recursive(container_type ops);
Expression handle_non_commuting(container_type ops, size_t index);
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

namespace {
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
