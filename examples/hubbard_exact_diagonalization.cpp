#include <armadillo>
#include <iostream>

#include "algebra/basis.h"
#include "algebra/hubbard_model.h"
#include "algebra/matrix_elements.h"

int main() {
  const size_t lattice_size = 4;
  const size_t particles = 2;
  const double hopping = 1.0;
  const double interaction = 4.0;

  // Build the model and a fixed-particle-number basis (spin is implicit).
  HubbardModel hubbard(hopping, interaction, lattice_size);
  Basis full_basis = Basis::with_fixed_particle_number(lattice_size, particles);
  Basis spin_zero_basis = Basis::with_fixed_particle_number_and_spin(lattice_size, particles, 0);

  // Construct the Hamiltonian matrix in the chosen basis.
  const Expression hamiltonian = hubbard.hamiltonian();
  arma::cx_mat H = compute_matrix_elements<arma::cx_mat>(full_basis, hamiltonian);
  arma::cx_mat H_spin_zero = compute_matrix_elements<arma::cx_mat>(spin_zero_basis, hamiltonian);

  // Exact diagonalization for a Hermitian Hamiltonian.
  arma::vec eigenvalues;
  arma::vec spin_zero_eigenvalues;
  arma::cx_mat eigenvectors;
  arma::cx_mat spin_zero_eigenvectors;
  if (!arma::eig_sym(eigenvalues, eigenvectors, H)) {
    std::cerr << "Diagonalization failed." << std::endl;
    return 1;
  }
  if (!arma::eig_sym(spin_zero_eigenvalues, spin_zero_eigenvectors, H_spin_zero)) {
    std::cerr << "Diagonalization failed for spin sector." << std::endl;
    return 1;
  }

  std::cout << "1D Hubbard model (L=4, N=2)" << std::endl;
  std::cout << "Full basis size: " << full_basis.set.size() << std::endl;
  std::cout << "Spin-0 basis size: " << spin_zero_basis.set.size() << std::endl;
  std::cout << "Ground state (full basis): " << eigenvalues(0) << std::endl;
  std::cout << "Ground state (spin 0): " << spin_zero_eigenvalues(0) << std::endl;

  return 0;
}
