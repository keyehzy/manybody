#include "algebra/hardcore_boson/expression.h"

HardcoreBosonExpression hopping(const HardcoreBosonExpression::complex_type& coeff, size_t from,
                                size_t to, HardcoreBosonOperator::Spin spin) noexcept {
  return hopping_generic<HardcoreBosonExpression>(coeff, from, to, spin);
}

HardcoreBosonExpression hopping(size_t from, size_t to,
                                HardcoreBosonOperator::Spin spin) noexcept {
  return hopping(1.0, from, to, spin);
}
