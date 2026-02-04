#include <iostream>

#include "algebra/basis.h"
#include "algebra/commutator.h"
#include "algebra/model/hubbard_model.h"
#include "algorithms/diagonal_terms.h"
#include "algorithms/schriffer_wolff.h"

int main() {
  const size_t lattice_size = 4;
  const size_t particles = 2;
  const double hopping = 1.0;
  const double interaction = -10.0;
  const size_t iterations = 200;

  HubbardModel hubbard(hopping, interaction, lattice_size);
  Basis basis = Basis::with_fixed_particle_number_and_spin(lattice_size, particles, 0);

  Expression kinetic = hubbard.kinetic();
  Expression interaction_term = hubbard.interaction();
  Expression hamiltonian = kinetic + interaction_term;

  Expression generator = schriffer_wolff(kinetic, interaction_term, basis, iterations);
  Expression effective_hamiltonian = BCH(generator, hamiltonian, 1.0, iterations);

  auto diagonal_terms = group_diagonal_children(effective_hamiltonian);

  std::cout << "1D Hubbard model (L=" << lattice_size << ", N=" << particles
            << ", U/t=" << interaction / hopping << ")\n";
  std::cout << "Basis size: " << basis.set.size() << "\n";
  std::cout << "Effective Hamiltonian (BCH):\n";

  for (const auto& term : diagonal_terms.diagonals) {
    std::cout << to_string(term) << "\n";
    for (const auto& child : diagonal_terms.children[term.operators]) {
      std::cout << "  " << to_string(child) << "\n";
    }
    std::cout << "\n";
  }

  return 0;
}
