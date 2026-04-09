#include "algebra/boson/expression.h"

#include <complex>

BosonExpression hopping(const BosonExpression::complex_type& coeff, size_t from, size_t to,
                        BosonOperator::Spin spin) noexcept {
  BosonExpression result = BosonExpression(BosonMonomial(
      coeff, {BosonOperator::creation(spin, from), BosonOperator::annihilation(spin, to)}));
  result +=
      BosonExpression(BosonMonomial(std::conj(coeff), {BosonOperator::creation(spin, to),
                                                       BosonOperator::annihilation(spin, from)}));
  return result;
}

BosonExpression hopping(size_t from, size_t to, BosonOperator::Spin spin) noexcept {
  return hopping(1.0, from, to, spin);
}
