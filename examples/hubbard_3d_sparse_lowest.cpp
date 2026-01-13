#include <armadillo>
#include <iostream>

#include "algebra/basis.h"
#include "algebra/hubbard_model.h"
#include "algebra/matrix_elements.h"

int main() {
  const size_t size_x = 2;
  const size_t size_y = 2;
  const size_t size_z = 2;
  const size_t sites = size_x * size_y * size_z;
  const size_t particles = 2;
  const double hopping = 1.0;
  const double interaction = 4.0;

  HubbardModel3D hubbard(hopping, interaction, size_x, size_y, size_z);
  Basis basis = Basis::with_fixed_particle_number_and_spin(sites, particles, 0);

  const Expression hamiltonian = hubbard.hamiltonian();
  arma::sp_cx_mat H = compute_matrix_elements<arma::sp_cx_mat>(basis, hamiltonian);

  arma::cx_vec eigenvalues;
  arma::cx_mat eigenvectors;
  if (!arma::eigs_gen(eigenvalues, eigenvectors, H, 1, "sr")) {
    std::cerr << "Sparse diagonalization failed." << std::endl;
    return 1;
  }

  std::cout << "3D Hubbard model (" << size_x << "x" << size_y << "x" << size_z
            << ", N=" << particles << ")" << std::endl;
  std::cout << "Spin-0 basis size: " << basis.set.size() << std::endl;
  std::cout << "Lowest eigenvalue (sparse): " << eigenvalues(0) << std::endl;

  return 0;
}
