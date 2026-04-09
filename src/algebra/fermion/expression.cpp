#include "algebra/fermion/expression.h"

#include <complex>

FermionExpression hopping(const FermionExpression::complex_type& coeff, size_t from, size_t to,
                          FermionOperator::Spin spin) noexcept {
  FermionExpression result = FermionExpression(FermionMonomial(
      coeff, {FermionOperator::creation(spin, from), FermionOperator::annihilation(spin, to)}));
  result += FermionExpression(FermionMonomial(
      std::conj(coeff),
      {FermionOperator::creation(spin, to), FermionOperator::annihilation(spin, from)}));
  return result;
}

FermionExpression hopping(size_t from, size_t to, FermionOperator::Spin spin) noexcept {
  return hopping(1.0, from, to, spin);
}
