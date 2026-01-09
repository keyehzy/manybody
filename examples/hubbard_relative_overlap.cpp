#include <armadillo>
#include <cstddef>
#include <iostream>

#include "hubbard_relative_operators.h"

int main() {
  const size_t lattice_size = 8;
  const size_t total_momentum = 0;
  const double t = 1.0;
  const double U = 4.0;
  const size_t num_steps = 10;

  HubbardRelativeKinetic kinetic(lattice_size, total_momentum);
  HubbardRelativeInteraction onsite(lattice_size);
  auto hamiltonian = t * kinetic + U * onsite;

  arma::vec v0(lattice_size, arma::fill::zeros);
  v0(0) = 1.0;

  arma::vec state = v0;
  for (size_t i = 0; i <= num_steps; ++i) {
    const double c_i = arma::dot(v0, state);
    std::cout << i << " " << c_i << "\n";
    state = hamiltonian.apply(state);
  }
  return 0;
}
