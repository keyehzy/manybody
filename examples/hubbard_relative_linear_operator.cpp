#include <iostream>

#include "hubbard_relative_operators.h"

int main() {
  const size_t lattice_size = 8;
  const int64_t total_momentum = 2;
  const double t = 1.0;
  const double U = 4.0;

  HubbardRelativeKinetic kinetic({lattice_size}, {total_momentum});
  HubbardRelative hamiltonian({lattice_size}, {total_momentum}, t, U);

  arma::cx_vec state(lattice_size, arma::fill::zeros);
  state(0) = 1.0;
  state(1) = 0.25;
  state(lattice_size - 1) = -0.5;

  arma::cx_vec result = hamiltonian.apply(state);

  std::cout << "Relative-coordinate Hubbard Hamiltonian\n";
  std::cout << "L=" << lattice_size << ", K=" << total_momentum
            << ", t_eff=" << kinetic.effective_hopping(0) * t << ", U=" << U << "\n";
  std::cout << "H|psi> = " << result.t();
  return 0;
}
