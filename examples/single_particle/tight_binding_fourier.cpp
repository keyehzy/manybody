#include <algorithm>
#include <cmath>
#include <complex>
#include <iomanip>
#include <iostream>
#include <numbers>
#include <vector>

#include "algebra/fermion/expression.h"
#include "algebra/fermion/term.h"
#include "algebra/fourier_transform.h"
#include "utils/index.h"

int main() {
  const size_t sites = 6;
  const double hopping = 1.0;
  Index index({sites});

  FermionExpression real_space;
  const FermionMonomial::complex_type coeff(-hopping, 0.0);
  for (size_t i = 0; i < sites; ++i) {
    const size_t j = (i + 1) % sites;
    real_space +=
        FermionMonomial(coeff, {FermionOperator::creation(FermionOperator::Spin::Up, i),
                                FermionOperator::annihilation(FermionOperator::Spin::Up, j)});
    real_space +=
        FermionMonomial(coeff, {FermionOperator::creation(FermionOperator::Spin::Up, j),
                                FermionOperator::annihilation(FermionOperator::Spin::Up, i)});
  }

  std::cout << "Real-space hopping Hamiltonian:\n" << real_space.to_string() << "\n";

  FermionExpression momentum_space = transform_expression(
      fourier_transform_operator<FermionExpression>, real_space, index, FourierMode::Direct);

  std::cout << "Momentum-space hopping Hamiltonian:\n" << momentum_space.to_string() << "\n";
  return 0;
}
