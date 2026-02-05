#include "algebra/commutator.h"

#include <cmath>

#include "utils/tolerances.h"

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

  return result;
}
