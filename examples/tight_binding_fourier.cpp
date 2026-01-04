#include <algorithm>
#include <cmath>
#include <complex>
#include <iomanip>
#include <iostream>
#include <numbers>
#include <vector>

#include "expression.h"
#include "fourier_transform.h"
#include "index.h"
#include "term.h"

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

  Expression momentum_space = transform_expression(
      [](Operator op, const DynamicIndex& idx) { return fourier_transform_operator(op, idx); },
      real_space, index);

  std::vector<std::complex<double>> diagonal(sites, {0.0, 0.0});
  double max_off_diagonal = 0.0;
  for (const auto& [ops, term_coeff] : momentum_space.hashmap) {
    if (ops.size() != 2) {
      continue;
    }
    if (ops[0].type() != Operator::Type::Creation ||
        ops[1].type() != Operator::Type::Annihilation) {
      continue;
    }
    if (ops[0].spin() != ops[1].spin()) {
      continue;
    }

    const size_t k = ops[0].value();
    const size_t k_prime = ops[1].value();
    const std::complex<double> coeff_double(term_coeff.real(), term_coeff.imag());
    if (k == k_prime) {
      diagonal[k] += coeff_double;
    } else {
      max_off_diagonal = std::max(max_off_diagonal, std::abs(coeff_double));
    }
  }

  std::cout << "1D tight-binding model (N=" << sites << ", t=" << hopping << ")\n";
  std::cout << "Diagonalized form in momentum space:\n";
  std::cout << std::setw(6) << "k" << std::setw(18) << "epsilon(k)" << std::setw(18) << "from FT"
            << "\n";

  std::cout.setf(std::ios::fixed);
  std::cout << std::setprecision(6);
  for (size_t k = 0; k < sites; ++k) {
    const double angle =
        2.0 * std::numbers::pi_v<double> * static_cast<double>(k) / static_cast<double>(sites);
    const double epsilon = -2.0 * hopping * std::cos(angle);
    std::cout << std::setw(6) << k << std::setw(18) << epsilon << std::setw(18)
              << diagonal[k].real() << "\n";
  }

  std::cout << "Max off-diagonal magnitude: " << max_off_diagonal << "\n";
  return 0;
}
