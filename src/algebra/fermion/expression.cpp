#include "algebra/fermion/expression.h"

#include <complex>

FermionExpression hopping(const FermionExpression::complex_type& coeff, size_t from, size_t to,
                          Operator::Spin spin) noexcept {
  FermionExpression result = FermionExpression(
      FermionMonomial(coeff, {Operator::creation(spin, from), Operator::annihilation(spin, to)}));
  result += FermionExpression(FermionMonomial(
      std::conj(coeff), {Operator::creation(spin, to), Operator::annihilation(spin, from)}));
  return result;
}

FermionExpression hopping(size_t from, size_t to, Operator::Spin spin) noexcept {
  return hopping(1.0, from, to, spin);
}
