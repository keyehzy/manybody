#include "commutator.h"

#include <cmath>
#include <limits>

#include "normal_order.h"

Expression commutator(const Term& A, const Term& B) {
  NormalOrderer orderer;
  Expression result = orderer.normal_order(A * B);
  result -= orderer.normal_order(B * A);
  return result;
}

Expression commutator(const Expression& A, const Expression& B) {
  NormalOrderer orderer;
  Expression result = orderer.normal_order(A * B);
  result -= orderer.normal_order(B * A);
  return result;
}

Expression anticommutator(const Term& A, const Term& B) {
  NormalOrderer orderer;
  Expression result = orderer.normal_order(A * B);
  result += orderer.normal_order(B * A);
  return result;
}

Expression anticommutator(const Expression& A, const Expression& B) {
  NormalOrderer orderer;
  Expression result = orderer.normal_order(A * B);
  result += orderer.normal_order(B * A);
  return result;
}

Expression BCH(const Expression& A, const Expression& B,
               Expression::complex_type::value_type lambda, size_t order) {
  NormalOrderer orderer;
  Expression current = B;
  Expression::complex_type::value_type coeff{1.0};
  Expression result = current * coeff;
  constexpr auto tolerance =
      1000.0 * std::numeric_limits<Expression::complex_type::value_type>::epsilon();

  for (size_t n = 1; n <= order; ++n) {
    current = commutator(A, current);
    coeff *= (lambda / static_cast<Expression::complex_type::value_type>(n));

    if (std::abs(coeff) < tolerance) {
      break;
    }

    result += current * coeff;
  }

  return orderer.normal_order(result);
}
