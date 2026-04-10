#include "algebra/boson/expression.h"

BosonExpression hopping(const BosonExpression::complex_type& coeff, size_t from, size_t to,
                        BosonOperator::Spin spin) noexcept {
  return hopping_generic<BosonExpression>(coeff, from, to, spin);
}

BosonExpression hopping(size_t from, size_t to, BosonOperator::Spin spin) noexcept {
  return hopping(1.0, from, to, spin);
}
