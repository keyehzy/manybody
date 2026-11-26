#include <armadillo>
#include <iostream>

#include "algebra/fermion/basis.h"
#include "algebra/fermion/model/hubbard_model.h"
#include "algebra/matrix_elements.h"
#include "algorithms/schriffer_wolff.h"

int main() {
  const size_t lattice_size = 15;
  const size_t particles = 2;
  const double hopping = 1.0;
  const double interaction = 100.0;
  const size_t iterations = 200;

  HubbardModel hubbard(hopping, interaction, lattice_size);
  FermionBasis basis =
      FermionBasis::with_fixed_particle_number_and_spin(lattice_size, particles, 0);

  FermionExpression kinetic = hubbard.kinetic();
  FermionExpression interaction_term = hubbard.interaction();
  FermionExpression hamiltonian = kinetic + interaction_term;

  FermionExpression generator = schriffer_wolff(kinetic, interaction_term, basis, iterations);
  FermionExpression effective_hamiltonian = BCH(generator, hamiltonian, 1.0, iterations);

  arma::cx_mat effective_matrix =
      compute_matrix_elements_serial<arma::cx_mat>(basis, effective_hamiltonian);

  arma::vec eigenvalues;
  arma::cx_mat eigenvectors;
  if (!arma::eig_sym(eigenvalues, eigenvectors, effective_matrix)) {
    std::cerr << "Diagonalization failed." << std::endl;
    return 1;
  }

  std::cout << "1D Hubbard model (L=" << lattice_size << ", N=" << particles
            << ", U/t=" << interaction / hopping << ")\n";
  std::cout << "FermionBasis size: " << basis.set.size() << "\n";
  std::cout << "Ground state energy (effective Hamiltonian): " << eigenvalues(0) << "\n";

  return 0;
}
