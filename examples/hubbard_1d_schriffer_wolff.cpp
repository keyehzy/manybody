#include <iostream>

#include "algorithm/schriffer_wolff.h"
#include "basis.h"
#include "commutator.h"
#include "models/hubbard_model.h"

int main() {
  const size_t lattice_size = 4;
  const size_t particles = 2;
  const double hopping = 1.0;
  const double interaction = 100.0;
  const size_t iterations = 200;

  HubbardModel hubbard(hopping, interaction, lattice_size);
  Basis basis = Basis::with_fixed_particle_number_and_spin(lattice_size, particles, 0);

  Expression kinetic = hubbard.kinetic();
  Expression interaction_term = hubbard.interaction();
  Expression hamiltonian = kinetic + interaction_term;

  Expression generator = schriffer_wolff(kinetic, interaction_term, basis, iterations);
  Expression effective_hamiltonian = BCH(generator, hamiltonian, 1.0, iterations);

  std::cout << "1D Hubbard model (L=" << lattice_size << ", N=" << particles
            << ", U/t=" << interaction / hopping << ")\n";
  std::cout << "Basis size: " << basis.set.size() << "\n";
  std::cout << "Effective Hamiltonian (BCH):\n";
  std::cout << effective_hamiltonian.to_string() << "\n";

  return 0;
}
