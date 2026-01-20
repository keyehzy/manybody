#include <armadillo>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <vector>

#include "algebra/basis.h"
#include "algebra/matrix_elements.h"
#include "algebra/model/hubbard_model_momentum.h"
#include "algebra/relative_basis_transform.h"
#include "utils/index.h"

constexpr double kTolerance = 1e-10;

// Test 1: Verify that the transformation matrix is unitary
bool test_unitarity(const arma::cx_mat& U) {
  const arma::cx_mat should_be_identity = U.t() * U;
  const arma::cx_mat identity = arma::eye<arma::cx_mat>(U.n_cols, U.n_cols);
  const double error = arma::norm(should_be_identity - identity, "fro");
  std::cout << "Unitarity test: ||U†U - I|| = " << error;
  if (error < kTolerance) {
    std::cout << " [PASS]\n";
    return true;
  } else {
    std::cout << " [FAIL]\n";
    return false;
  }
}

// Test 2: Verify eigenvalues are preserved under transformation
bool test_eigenvalue_preservation(const arma::cx_mat& H_mom, const arma::cx_mat& U) {
  // Transform Hamiltonian to relative position basis
  const arma::cx_mat H_rel = U.t() * H_mom * U;

  // Get eigenvalues of both matrices
  arma::vec eig_mom;
  arma::eig_sym(eig_mom, arma::cx_mat(H_mom));

  arma::vec eig_rel;
  arma::eig_sym(eig_rel, arma::cx_mat(H_rel));

  // Sort and compare
  eig_mom = arma::sort(eig_mom);
  eig_rel = arma::sort(eig_rel);

  const double error = arma::norm(eig_mom - eig_rel);
  std::cout << "Eigenvalue preservation test: ||λ_mom - λ_rel|| = " << error;
  if (error < kTolerance) {
    std::cout << " [PASS]\n";
    return true;
  } else {
    std::cout << " [FAIL]\n";
    return false;
  }
}

// Test 3: Verify interaction localizes to r=0
bool test_interaction_localization(const arma::cx_mat& H_interaction_rel, double U_val) {
  // The interaction should be U at (0,0) and ~0 elsewhere
  const double diag_value = std::abs(H_interaction_rel(0, 0));
  double off_diag_max = 0.0;
  for (size_t i = 0; i < H_interaction_rel.n_rows; ++i) {
    for (size_t j = 0; j < H_interaction_rel.n_cols; ++j) {
      if (i == 0 && j == 0) continue;
      off_diag_max = std::max(off_diag_max, std::abs(H_interaction_rel(i, j)));
    }
  }

  const bool diag_correct = std::abs(diag_value - U_val) < kTolerance;
  const bool off_diag_correct = off_diag_max < kTolerance;

  std::cout << "Interaction localization test: H_int(0,0)=" << diag_value << " (expected " << U_val
            << "), max off-diag=" << off_diag_max;
  if (diag_correct && off_diag_correct) {
    std::cout << " [PASS]\n";
    return true;
  } else {
    std::cout << " [FAIL]\n";
    return false;
  }
}

int main() {
  std::cout << "=== Testing Relative Basis Transform ===\n\n";

  bool all_passed = true;

  // Test parameters
  const size_t lattice_size = 4;
  const std::vector<size_t> size{lattice_size};
  const int64_t K = 1;  // Total momentum
  const double t = 1.0;
  const double U = 4.0;

  std::cout << "Parameters: L=" << lattice_size << ", K=" << K << ", t=" << t << ", U=" << U
            << "\n\n";

  // Build momentum basis for 2 particles (1 up, 1 down) with fixed total momentum
  Index index(size);
  Basis momentum_basis = Basis::with_fixed_particle_number_spin_momentum(
      lattice_size, 2, 0,  // 2 particles, spin projection 0 (1 up, 1 down)
      index, {static_cast<size_t>(K)});

  std::cout << "Momentum basis size: " << momentum_basis.set.size() << "\n\n";

  // Build Hamiltonian in momentum basis
  HubbardModelMomentum hubbard(t, U, size);
  const Expression H_expr = hubbard.hamiltonian();
  arma::cx_mat H_mom = compute_matrix_elements<arma::cx_mat>(momentum_basis, H_expr);

  // Build transformation matrix
  arma::cx_mat U_transform = relative_position_transform(momentum_basis, index);

  // Run tests
  all_passed &= test_unitarity(U_transform);
  all_passed &= test_eigenvalue_preservation(H_mom, U_transform);

  // Test interaction localization
  arma::cx_mat H_interaction_mom =
      compute_matrix_elements<arma::cx_mat>(momentum_basis, hubbard.interaction());
  arma::cx_mat H_interaction_rel = U_transform.t() * H_interaction_mom * U_transform;
  all_passed &= test_interaction_localization(H_interaction_rel, U);

  std::cout << "\n=== " << (all_passed ? "All Tests Passed" : "Some Tests Failed") << " ===\n";
  return all_passed ? 0 : 1;
}
