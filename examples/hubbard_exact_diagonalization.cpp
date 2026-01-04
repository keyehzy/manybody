#include <armadillo>
#include <iostream>

#include "basis.h"
#include "matrix_elements.h"
#include "models/hubbard_model.h"

int main() {
  const size_t lattice_size = 4;
  const size_t particles = 2;
  const double hopping = 1.0;
  const double interaction = 4.0;

  // Build the model and a fixed-particle-number basis (spin is implicit).
  HubbardModel hubbard(hopping, interaction, lattice_size);
  Basis basis = Basis::with_fixed_particle_number(lattice_size, particles);

  // Construct the Hamiltonian matrix in the chosen basis.
  const Expression hamiltonian = hubbard.hamiltonian();
  arma::cx_mat H = compute_matrix_elements<arma::cx_mat>(basis, hamiltonian);

  // Exact diagonalization for a Hermitian Hamiltonian.
  arma::vec eigenvalues;
  arma::cx_mat eigenvectors;
  if (!arma::eig_sym(eigenvalues, eigenvectors, H)) {
    std::cerr << "Diagonalization failed." << std::endl;
    return 1;
  }

  std::cout << "1D Hubbard model (L=4, N=2)" << std::endl;
  std::cout << "Basis size: " << basis.set.size() << std::endl;
  std::cout << "Eigenvalues:" << std::endl;
  for (size_t i = 0; i < eigenvalues.n_elem; ++i) {
    std::cout << "  E[" << i << "] = " << eigenvalues(i) << std::endl;
  }

  return 0;
}
