#pragma once

#include <complex>
#include <cstddef>
#include <utility>

#include "algebra/statistics.h"
#include "utils/static_vector.h"
#include "utils/tolerances.h"

namespace detail {

template <typename Expression>
Expression canonicalize_recursive(typename Expression::container_type ops);

template <typename Expression>
Expression handle_non_commuting(typename Expression::container_type ops, size_t index);

template <typename Expression>
Expression canonicalize_recursive(typename Expression::container_type ops) {
  using Traits = AlgebraTraits<Expression::operator_type::statistics>;
  using complex_type = typename Expression::complex_type;

  if (ops.size() < 2) {
    return Expression(ops);
  }

  if constexpr (Traits::pauli_exclusion) {
    if (has_consecutive_elements(ops)) {
      return {};
    }
  }

  complex_type phase{1.0, 0.0};

  for (size_t i = 1; i < ops.size(); ++i) {
    size_t j = i;
    while (j > 0 && ops[j] < ops[j - 1]) {
      if (ops[j].commutes(ops[j - 1])) {
        std::swap(ops[j], ops[j - 1]);
        if constexpr (Traits::swap_sign == -1) {
          phase = -phase;
        }
        --j;
      } else {
        Expression result = handle_non_commuting<Expression>(std::move(ops), j - 1);
        result *= phase;
        return result;
      }
    }
  }

  if constexpr (Traits::pauli_exclusion) {
    if (has_consecutive_elements(ops)) {
      return {};
    }
  }

  return Expression(phase, std::move(ops));
}

template <typename Expression>
Expression handle_non_commuting(typename Expression::container_type ops, size_t index) {
  using Traits = AlgebraTraits<Expression::operator_type::statistics>;
  using container_type = typename Expression::container_type;

  container_type contracted;
  contracted.append_range(ops.begin(), ops.begin() + index);
  contracted.append_range(ops.begin() + index + 2, ops.end());
  std::swap(ops[index], ops[index + 1]);
  Expression lhs = canonicalize_recursive<Expression>(std::move(contracted));
  if constexpr (Traits::contraction_sign == +1) {
    lhs += canonicalize_recursive<Expression>(std::move(ops));
  } else {
    lhs -= canonicalize_recursive<Expression>(std::move(ops));
  }
  return lhs;
}

}  // namespace detail

template <typename Expression>
Expression canonicalize_generic(const typename Expression::complex_type& c,
                                const typename Expression::container_type& ops) {
  using complex_type = typename Expression::complex_type;
  constexpr auto tolerance = tolerances::tolerance<typename complex_type::value_type>();
  if (std::norm(c) < tolerance * tolerance) {
    return {};
  }
  Expression result = detail::canonicalize_recursive<Expression>(ops);
  result *= c;
  return result;
}

template <typename Expression>
Expression canonicalize_generic(const Expression& expr) {
  Expression result;
  for (const auto& [ops, c] : expr.terms()) {
    result += canonicalize_generic<Expression>(c, ops);
  }
  return result;
}
