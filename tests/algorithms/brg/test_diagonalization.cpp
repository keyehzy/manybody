#include <catch2/catch.hpp>

#include "algebra/fermion/basis.h"
#include "algebra/fermion/expression.h"
#include "algorithms/brg/diagonalization.h"

TEST_CASE("diagonalize_sector returns correct number of eigenvalues") {
  // Single-site, single particle sector (N=1, Sz=1): dimension 1
  Basis basis = Basis::with_fixed_particle_number_and_spin(1, 1, 1);

  // Simple Hamiltonian: H = -mu * n
  double mu = 0.5;
  Expression H = Expression(density(Operator::Spin::Up, 0)) * std::complex<double>(-mu, 0.0);

  auto result = brg::diagonalize_sector(basis, H);

  CHECK(result.eigenvalues.n_elem == 1);
  CHECK(result.eigenvectors.n_rows == 1);
  CHECK(result.eigenvectors.n_cols == 1);
  CHECK(result.eigenvalues(0) == Approx(-mu).margin(1e-12));
}

TEST_CASE("diagonalize_sector with two-site hopping") {
  // Two sites, one up-spin electron: dimension 2
  Basis basis = Basis::with_fixed_particle_number_and_spin(2, 1, 1);

  // Hopping Hamiltonian: H = -t (c†_0 c_1 + c†_1 c_0)
  double t = 1.0;
  Expression H = hopping({-t, 0.0}, 0, 1, Operator::Spin::Up);

  auto result = brg::diagonalize_sector(basis, H);

  // Eigenvalues should be ±t
  CHECK(result.eigenvalues.n_elem == 2);
  CHECK(result.eigenvalues(0) == Approx(-t).margin(1e-12));
  CHECK(result.eigenvalues(1) == Approx(t).margin(1e-12));
}

TEST_CASE("diagonalize_sector eigenvalues are sorted ascending") {
  // Two-site Hubbard with one electron
  Basis basis = Basis::with_fixed_particle_number_and_spin(2, 1, 1);

  double t = 2.0;
  Expression H = hopping({-t, 0.0}, 0, 1, Operator::Spin::Up);

  auto result = brg::diagonalize_sector(basis, H);

  for (size_t i = 1; i < result.eigenvalues.n_elem; ++i) {
    CHECK(result.eigenvalues(i) >= result.eigenvalues(i - 1));
  }
}

TEST_CASE("diagonalize_sector eigenvectors are orthonormal") {
  Basis basis = Basis::with_fixed_particle_number_and_spin(2, 1, 1);

  double t = 1.5;
  Expression H = hopping({-t, 0.0}, 0, 1, Operator::Spin::Up);

  auto result = brg::diagonalize_sector(basis, H);

  // Check orthonormality
  arma::cx_mat overlap = result.eigenvectors.t() * result.eigenvectors;
  arma::cx_mat identity = arma::eye<arma::cx_mat>(2, 2);

  for (size_t i = 0; i < 2; ++i) {
    for (size_t j = 0; j < 2; ++j) {
      CHECK(std::abs(overlap(i, j) - identity(i, j)) < 1e-12);
    }
  }
}

TEST_CASE("diagonalize_sector vacuum sector has single zero eigenvalue") {
  // Vacuum sector (N=0, Sz=0): dimension 1
  Basis basis = Basis::with_fixed_particle_number_and_spin(2, 0, 0);

  // Any Hamiltonian should give E=0 for vacuum
  double t = 1.0;
  double U = -4.0;
  Expression H = hopping({-t, 0.0}, 0, 1, Operator::Spin::Up);
  H += Expression(density_density(Operator::Spin::Up, 0, Operator::Spin::Down, 0)) *
       std::complex<double>(U, 0.0);

  auto result = brg::diagonalize_sector(basis, H);

  CHECK(result.eigenvalues.n_elem == 1);
  CHECK(result.eigenvalues(0) == Approx(0.0).margin(1e-12));
}
