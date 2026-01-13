#include <algorithm>
#include <cmath>
#include <complex>
#include <iomanip>
#include <iostream>
#include <numbers>
#include <vector>

#include "algebra/expression.h"
#include "algebra/fourier_transform.h"
#include "algebra/term.h"
#include "utils/index.h"

int main() {
  const size_t sites = 6;
  const double hopping = 1.0;
  DynamicIndex index({sites});

  Expression real_space;
  const Term::complex_type coeff(static_cast<float>(-hopping), 0.0f);
  for (size_t i = 0; i < sites; ++i) {
    const size_t j = (i + 1) % sites;
    real_space += Term(coeff, {Operator::creation(Operator::Spin::Up, i),
                               Operator::annihilation(Operator::Spin::Up, j)});
    real_space += Term(coeff, {Operator::creation(Operator::Spin::Up, j),
                               Operator::annihilation(Operator::Spin::Up, i)});
  }

  std::cout << "Real-space hopping Hamiltonian:\n" << real_space.to_string() << "\n";

  Expression momentum_space = transform_expression(fourier_transform_operator, real_space, index);

  std::cout << "Momentum-space hopping Hamiltonian:\n" << momentum_space.to_string() << "\n";
  return 0;
}
