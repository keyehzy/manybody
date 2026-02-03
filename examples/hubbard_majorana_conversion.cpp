#include <cstddef>
#include <iostream>

#include "algebra/majorana/conversion.h"
#include "algebra/model/hubbard_model.h"

using namespace majorana;

int main() {
  const double t = 1.0;
  const double u = 4.0;
  const size_t sites = 2;

  HubbardModel hubbard(t, u, sites);
  const Expression hamiltonian = hubbard.hamiltonian();
  const MajoranaExpression majorana_hamiltonian = to_majorana(hamiltonian);

  std::cout << "=== Hubbard Model to Majorana Example ===\n";
  std::cout << "Parameters: L=" << sites << ", t=" << t << ", U=" << u << "\n\n";
  std::cout << "Fermionic Hamiltonian (terms=" << hamiltonian.size() << "):\n";
  std::cout << hamiltonian.to_string() << "\n\n";
  std::cout << "Majorana Hamiltonian (terms=" << majorana_hamiltonian.size() << "):\n";
  std::cout << majorana_hamiltonian.to_string() << "\n";

  return 0;
}
