#include <armadillo>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <vector>

#include "algebra/basis.h"
#include "algebra/matrix_elements.h"
#include "algebra/model/hubbard_model_momentum.h"
#include "algebra/relative_basis_transform.h"
#include "utils/index.h"

// Demonstrates the relative position basis transformation for the 2-particle Hubbard model.
//
// The transformation converts from momentum basis |p_up, K-p_up> to relative position basis |r>.
// Key properties:
// - The transformation matrix U is unitary (preserves eigenvalues)
// - The on-site interaction localizes to r=0 in the relative basis
// - The kinetic energy becomes diagonal in momentum space but spreads in position space
int main() {
  const size_t lattice_size = 6;
  const std::vector<size_t> size{lattice_size};
  const size_t K = 2;  // Total momentum
  const double t = 1.0;
  const double U = 4.0;

  std::cout << "=== Relative Basis Transform Example ===\n\n";
  std::cout << "Parameters: L=" << lattice_size << ", K=" << K << ", t=" << t << ", U=" << U
            << "\n\n";

  // Build momentum basis for 2 particles (1 up, 1 down) with fixed total momentum
  Index index(size);
  Basis momentum_basis =
      Basis::with_fixed_particle_number_spin_momentum(lattice_size, 2, 0, index, {K});

  std::cout << "Momentum basis states (|p_up, p_down>):\n";
  for (size_t i = 0; i < momentum_basis.set.size(); ++i) {
    const auto& state = momentum_basis.set[i];
    std::cout << "  " << i << ": |" << state[0].value() << ", " << state[1].value() << ">\n";
  }
  std::cout << "\n";

  // Build Hubbard Hamiltonian in momentum basis
  HubbardModelMomentum hubbard(t, U, size);
  arma::cx_mat H_mom = compute_matrix_elements<arma::cx_mat>(momentum_basis, hubbard.hamiltonian());

  std::cout << "Hamiltonian in momentum basis:\n" << arma::real(H_mom) << "\n";

  // Build transformation matrix
  auto result = relative_position_transform_with_index(momentum_basis, index);
  const arma::cx_mat& U_transform = result.transform;

  std::cout << "Transformation matrix (real part):\n" << arma::real(U_transform) << "\n";
  std::cout << "Transformation matrix (imag part):\n" << arma::imag(U_transform) << "\n";

  // Transform Hamiltonian to relative position basis
  const arma::cx_mat H_rel = U_transform.t() * H_mom * U_transform;

  std::cout << "Hamiltonian in relative position basis:\n" << arma::real(H_rel) << "\n";

  // Show that interaction localizes to r=0
  arma::cx_mat H_int_mom =
      compute_matrix_elements<arma::cx_mat>(momentum_basis, hubbard.interaction());
  arma::cx_mat H_int_rel = U_transform.t() * H_int_mom * U_transform;

  std::cout << "Interaction in momentum basis (real part):\n" << arma::real(H_int_mom) << "\n";
  std::cout << "Interaction in relative basis (real part):\n" << arma::real(H_int_rel) << "\n";
  std::cout << "Note: Interaction localizes to H_int(0,0) = " << std::real(H_int_rel(0, 0))
            << " (r=0)\n\n";

  // Compare eigenvalues
  arma::vec eig_mom;
  arma::eig_sym(eig_mom, arma::cx_mat(H_mom));

  arma::vec eig_rel;
  arma::eig_sym(eig_rel, arma::cx_mat(H_rel));

  std::cout << std::fixed << std::setprecision(6);
  std::cout << "Eigenvalues (momentum basis): " << eig_mom.t();
  std::cout << "Eigenvalues (relative basis): " << eig_rel.t();
  std::cout << "Difference norm: " << arma::norm(eig_mom - eig_rel) << "\n";

  return 0;
}
