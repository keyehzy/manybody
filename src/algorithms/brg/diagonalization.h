#pragma once

#include <armadillo>
#include <cstdlib>
#include <iostream>

#include "algebra/fermion/basis.h"
#include "algebra/fermion/expression.h"
#include "algebra/fermion/matrix_elements.h"

namespace brg {

/// Result of diagonalizing a symmetry sector.
struct SectorResult {
  arma::vec eigenvalues;
  arma::cx_mat eigenvectors;
};

/// Diagonalize the Hamiltonian in a given symmetry sector.
/// Returns eigenvalues (ascending) and eigenvectors.
inline SectorResult diagonalize_sector(const Basis& basis, const Expression& H) {
  arma::cx_mat mat = compute_matrix_elements<arma::cx_mat>(basis, H);

  SectorResult result;
  if (!arma::eig_sym(result.eigenvalues, result.eigenvectors, mat)) {
    std::cerr << "Diagonalization failed.\n";
    std::exit(1);
  }
  return result;
}

}  // namespace brg
