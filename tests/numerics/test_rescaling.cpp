#include <armadillo>
#include <catch2/catch.hpp>
#include <cmath>

#include "numerics/rescaling.h"

namespace test_rescaling {

TEST_CASE("rescaling_rescale_and_inverse_are_inverses") {
  rescaling::Rescaling r{2.5, 1.0};

  const double E = 3.5;
  const double E_scaled = r.rescale(E);
  const double E_recovered = r.inverse(E_scaled);

  CHECK(std::abs(E_recovered - E) < 1e-14);
}

TEST_CASE("rescaling_rescale_maps_to_expected_values") {
  // Spectrum from -1 to 5, so a = 3, b = 2
  rescaling::Rescaling r{3.0, 2.0};

  // E = b should map to 0
  CHECK(std::abs(r.rescale(2.0)) < 1e-14);

  // E = b + a should map to 1
  CHECK(std::abs(r.rescale(5.0) - 1.0) < 1e-14);

  // E = b - a should map to -1
  CHECK(std::abs(r.rescale(-1.0) + 1.0) < 1e-14);
}

TEST_CASE("rescaling_inverse_maps_from_expected_values") {
  rescaling::Rescaling r{3.0, 2.0};

  // 0 maps to b
  CHECK(std::abs(r.inverse(0.0) - 2.0) < 1e-14);

  // 1 maps to b + a
  CHECK(std::abs(r.inverse(1.0) - 5.0) < 1e-14);

  // -1 maps to b - a
  CHECK(std::abs(r.inverse(-1.0) + 1.0) < 1e-14);
}

TEST_CASE("rescaling_estimate_rescaling_puts_spectrum_in_range") {
  arma::mat H(4, 4, arma::fill::zeros);
  H.diag() = arma::vec{-2.0, -1.0, 1.0, 3.0};

  auto r = rescaling::estimate_rescaling(H);
  arma::mat H_scaled = rescaling::rescale_hamiltonian(H, r);

  arma::vec eigvals = arma::eig_sym(H_scaled);

  // All eigenvalues should be strictly in (-1, 1) due to padding
  CHECK(eigvals.min() > -1.0);
  CHECK(eigvals.max() < 1.0);
}

TEST_CASE("rescaling_estimate_rescaling_with_padding") {
  arma::mat H(3, 3, arma::fill::zeros);
  H.diag() = arma::vec{0.0, 1.0, 2.0};

  // With zero padding, spectrum should reach exactly [-1, 1]
  // Note: padding=0 would put eigenvalues exactly at boundaries
  auto r_no_pad = rescaling::estimate_rescaling(H, 0.0);
  CHECK(std::abs(r_no_pad.a - 1.0) < 1e-10);  // (2-0)/2 = 1
  CHECK(std::abs(r_no_pad.b - 1.0) < 1e-10);  // (2+0)/2 = 1

  // With padding, scale should be larger
  auto r_padded = rescaling::estimate_rescaling(H, 0.1);
  CHECK(r_padded.a > r_no_pad.a);
  CHECK(std::abs(r_padded.b - 1.0) < 1e-10);  // Center shouldn't change
}

TEST_CASE("rescaling_from_bounds_creates_correct_rescaling") {
  auto r = rescaling::from_bounds(-2.0, 4.0, 0.0);

  // a = (4 - (-2)) / 2 = 3
  // b = (4 + (-2)) / 2 = 1
  CHECK(std::abs(r.a - 3.0) < 1e-14);
  CHECK(std::abs(r.b - 1.0) < 1e-14);
}

TEST_CASE("rescaling_from_bounds_with_padding") {
  auto r = rescaling::from_bounds(-2.0, 4.0, 0.1);

  // a = (4 - (-2)) / 2 * 1.1 = 3.3
  // b = (4 + (-2)) / 2 = 1
  CHECK(std::abs(r.a - 3.3) < 1e-14);
  CHECK(std::abs(r.b - 1.0) < 1e-14);
}

TEST_CASE("rescaling_rescale_hamiltonian_preserves_eigenvectors") {
  arma::mat H(3, 3, arma::fill::zeros);
  H.diag() = arma::vec{-1.0, 0.0, 2.0};
  H(0, 1) = 0.5;
  H(1, 0) = 0.5;

  arma::vec eigvals_orig;
  arma::mat eigvecs_orig;
  arma::eig_sym(eigvals_orig, eigvecs_orig, H);

  auto r = rescaling::estimate_rescaling(H);
  arma::mat H_scaled = rescaling::rescale_hamiltonian(H, r);

  arma::vec eigvals_scaled;
  arma::mat eigvecs_scaled;
  arma::eig_sym(eigvals_scaled, eigvecs_scaled, H_scaled);

  // Eigenvectors should be the same (up to sign)
  for (size_t i = 0; i < 3; ++i) {
    double dot = std::abs(arma::dot(eigvecs_orig.col(i), eigvecs_scaled.col(i)));
    CHECK(std::abs(dot - 1.0) < 1e-10);
  }
}

TEST_CASE("rescaling_rescale_hamiltonian_sparse_matches_dense") {
  arma::mat H_dense(4, 4, arma::fill::zeros);
  H_dense.diag() = arma::vec{-2.0, -1.0, 1.0, 3.0};
  H_dense(0, 1) = 0.5;
  H_dense(1, 0) = 0.5;

  arma::sp_mat H_sparse(H_dense);

  auto r = rescaling::from_bounds(-2.5, 3.5);

  arma::mat H_scaled_dense = rescaling::rescale_hamiltonian(H_dense, r);
  arma::sp_mat H_scaled_sparse = rescaling::rescale_hamiltonian(H_sparse, r);

  arma::mat diff = H_scaled_dense - arma::mat(H_scaled_sparse);
  CHECK(arma::norm(diff, "fro") < 1e-14);
}

}  // namespace test_rescaling
