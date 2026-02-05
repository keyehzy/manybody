#include "algebra/commutator.h"

#include <cmath>

#include "algebra/normal_order.h"
#include "utils/tolerances.h"

Expression commutator(const FermionMonomial& A, const FermionMonomial& B) {
  Expression result = normal_order(A * B);
  result -= normal_order(B * A);
  return result;
}

Expression commutator(const Expression& A, const Expression& B) {
  Expression result = normal_order(A * B);
  result -= normal_order(B * A);
  return result;
}

Expression anticommutator(const FermionMonomial& A, const FermionMonomial& B) {
  Expression result = normal_order(A * B);
  result += normal_order(B * A);
  return result;
}

Expression anticommutator(const Expression& A, const Expression& B) {
  Expression result = normal_order(A * B);
  result += normal_order(B * A);
  return result;
}

Expression BCH(const Expression& A, const Expression& B,
               Expression::complex_type::value_type lambda, size_t order) {
  Expression current = B;
  Expression::complex_type::value_type coeff{1.0};
  Expression result = current * coeff;
  constexpr auto tolerance = tolerances::tolerance<Expression::complex_type::value_type>();

  for (size_t n = 1; n <= order; ++n) {
    current = commutator(A, current);
    coeff *= (lambda / static_cast<Expression::complex_type::value_type>(n));

    if (std::abs(coeff) < tolerance) {
      break;
    }

    result += current * coeff;
  }

  return normal_order(result);
}
