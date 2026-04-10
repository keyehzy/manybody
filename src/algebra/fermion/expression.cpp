#include "algebra/fermion/expression.h"

FermionExpression hopping(const FermionExpression::complex_type& coeff, size_t from, size_t to,
                          FermionOperator::Spin spin) noexcept {
  return hopping_generic<FermionExpression>(coeff, from, to, spin);
}

FermionExpression hopping(size_t from, size_t to, FermionOperator::Spin spin) noexcept {
  return hopping(1.0, from, to, spin);
}
