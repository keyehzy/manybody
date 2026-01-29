#pragma once

#include <armadillo>

namespace rescaling {

/// Rescaling parameters for mapping Hamiltonian spectrum to [-1, 1]
/// H_scaled = (H - b) / a, where a = (E_max - E_min) / 2, b = (E_max + E_min) / 2
struct Rescaling {
  double a;  // Scale factor
  double b;  // Shift

  /// Rescale an energy value
  double rescale(double E) const { return (E - b) / a; }

  /// Inverse rescale
  double inverse(double E_scaled) const { return E_scaled * a + b; }
};

/// Estimate spectral bounds using exact diagonalization (dense matrices)
inline Rescaling estimate_rescaling(const arma::mat& H, double padding = 0.01) {
  arma::vec eigvals = arma::eig_sym(H);
  const double E_min = eigvals.min();
  const double E_max = eigvals.max();

  const double a = (E_max - E_min) / 2.0 * (1.0 + padding);
  const double b = (E_max + E_min) / 2.0;

  return Rescaling{a, b};
}

/// Create rescaling from known spectral bounds
inline Rescaling from_bounds(double E_min, double E_max, double padding = 0.01) {
  const double a = (E_max - E_min) / 2.0 * (1.0 + padding);
  const double b = (E_max + E_min) / 2.0;

  return Rescaling{a, b};
}

/// Rescale dense matrix
inline arma::mat rescale_hamiltonian(const arma::mat& H, const Rescaling& rescaling) {
  return (H - rescaling.b * arma::eye(H.n_rows, H.n_cols)) / rescaling.a;
}

/// Rescale sparse matrix
inline arma::sp_mat rescale_hamiltonian(const arma::sp_mat& H, const Rescaling& rescaling) {
  return (H - rescaling.b * arma::speye<arma::sp_mat>(H.n_rows, H.n_cols)) / rescaling.a;
}

}  // namespace rescaling
