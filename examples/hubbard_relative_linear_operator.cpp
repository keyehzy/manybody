#include <iostream>

#include "hubbard_relative_operators.h"

int main() {
  const size_t lattice_size = 8;
  const size_t total_momentum = 2;
  const double t = 1.0;
  const double U = 4.0;

  HubbardRelativeKinetic kinetic({lattice_size}, {total_momentum});
  HubbardRelativeInteraction onsite(lattice_size);

  auto hamiltonian = t * kinetic + U * onsite;

  arma::vec state(lattice_size, arma::fill::zeros);
  state(0) = 1.0;
  state(1) = 0.25;
  state(lattice_size - 1) = -0.5;

  arma::vec result = hamiltonian.apply(state);

  std::cout << "Relative-coordinate Hubbard Hamiltonian\n";
  std::cout << "L=" << lattice_size << ", K=" << total_momentum
            << ", t_eff=" << kinetic.effective_hopping(0) * t << ", U=" << U << "\n";
  std::cout << "H|psi> = " << result.t();
  return 0;
}
