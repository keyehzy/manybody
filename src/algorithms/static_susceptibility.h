#pragma once

#include <armadillo>
#include <complex>
#include <cstddef>
#include <stdexcept>

struct StaticSusceptibilityResult {
  double value{0.0};
  double ground_energy{0.0};
  size_t skipped_states{0};
};

inline StaticSusceptibilityResult compute_zero_temperature_static_susceptibility(
    const arma::vec& eigenvalues, const arma::cx_mat& eigenvectors, const arma::cx_mat& observable,
    double gap_tolerance = 1e-12) {
  if (eigenvalues.n_elem == 0) {
    throw std::invalid_argument("eigenvalues must not be empty.");
  }
  if (eigenvectors.n_cols != eigenvalues.n_elem || eigenvectors.n_rows != observable.n_rows ||
      observable.n_rows != observable.n_cols || observable.n_cols != eigenvectors.n_rows) {
    throw std::invalid_argument("incompatible eigenvector, eigenvalue, and observable dimensions.");
  }
  if (gap_tolerance < 0.0) {
    throw std::invalid_argument("gap_tolerance must be non-negative.");
  }

  StaticSusceptibilityResult result;
  result.ground_energy = eigenvalues(0);

  const arma::cx_vec ground_state = eigenvectors.col(0);
  const arma::cx_vec observable_ground_state = observable * ground_state;

  for (size_t n = 1; n < eigenvalues.n_elem; ++n) {
    const double gap = eigenvalues(n) - result.ground_energy;
    if (gap <= gap_tolerance) {
      ++result.skipped_states;
      continue;
    }

    const std::complex<double> matrix_element =
        arma::cdot(eigenvectors.col(n), observable_ground_state);
    result.value += std::norm(matrix_element) / gap;
  }

  return result;
}
