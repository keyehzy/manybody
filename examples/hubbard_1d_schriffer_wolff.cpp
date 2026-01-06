#include <armadillo>
#include <iostream>

#include "algorithm/schriffer_wolff.h"
#include "basis.h"
#include "matrix_elements.h"
#include "models/hubbard_model.h"

int main() {
  const size_t lattice_size = 4;
  const size_t particles = 4;
  const double hopping = 1.0;
  const double interaction = 12.0;
  const size_t iterations = 200;

  HubbardModel hubbard(hopping, interaction, lattice_size);
  Basis basis = Basis::with_fixed_particle_number_and_spin(lattice_size, particles, 0);

  Expression kinetic = hubbard.kinetic();
  Expression interaction_term = hubbard.interaction();
  Expression hamiltonian = kinetic + interaction_term;

  arma::cx_mat H = compute_matrix_elements<arma::cx_mat>(basis, hamiltonian);
  arma::cx_mat H_interaction = compute_matrix_elements<arma::cx_mat>(basis, interaction_term);

  arma::vec interaction_eigs;
  arma::cx_mat interaction_vecs;
  if (!arma::eig_sym(interaction_eigs, interaction_vecs, H_interaction)) {
    std::cerr << "Diagonalization failed." << std::endl;
    return 1;
  }

  const size_t split = cluster_by_largest_gap(interaction_eigs).first;
  const auto off_diagonal_norm = [&](const arma::cx_mat& mat) {
    if (split == 0 || split >= mat.n_rows) {
      return 0.0;
    }
    return arma::norm(mat.submat(0, split, split - 1, mat.n_cols - 1), "fro");
  };

  arma::cx_mat H_in = interaction_vecs.t() * H * interaction_vecs;
  const double before_norm = off_diagonal_norm(H_in);

  Expression generator = schriffer_wolff(kinetic, interaction_term, basis, iterations);
  arma::cx_mat A = compute_matrix_elements<arma::cx_mat>(basis, generator);
  arma::cx_mat U = arma::expmat(A);
  arma::cx_mat H_eff = U.t() * H * U;
  arma::cx_mat H_eff_in = interaction_vecs.t() * H_eff * interaction_vecs;
  const double after_norm = off_diagonal_norm(H_eff_in);

  std::cout << "1D Hubbard model (L=4, N=4, U/t=" << interaction / hopping << ")\n";
  std::cout << "Basis size: " << basis.set.size() << "\n";
  std::cout << "Interaction gap split index: " << split << "\n";
  std::cout << "Off-diagonal norm before SW: " << before_norm << "\n";
  std::cout << "Off-diagonal norm after SW: " << after_norm << "\n";

  return 0;
}
