#include <armadillo>
#include <catch2/catch.hpp>

#include "algebra/basis.h"
#include "algebra/matrix_elements.h"
#include "algorithms/brg/hubbard_block.h"

TEST_CASE("block_2d_2x2 geometry is correct") {
  auto geom = brg::block_2d_2x2();

  CHECK(geom.num_sites == 4);
  CHECK(geom.bonds.size() == 4);
  CHECK(geom.border_sites.size() == 2);
  CHECK(geom.nu == 2.0);

  // Check border sites are 1 and 3
  CHECK(geom.border_sites[0] == 1);
  CHECK(geom.border_sites[1] == 3);
}

TEST_CASE("block_3d_2x2x2 geometry is correct") {
  auto geom = brg::block_3d_2x2x2();

  CHECK(geom.num_sites == 8);
  CHECK(geom.bonds.size() == 12);
  CHECK(geom.border_sites.size() == 4);
  CHECK(geom.nu == 4.0);

  // Check border sites are 1, 3, 5, 7 (right +x face)
  CHECK(geom.border_sites[0] == 1);
  CHECK(geom.border_sites[1] == 3);
  CHECK(geom.border_sites[2] == 5);
  CHECK(geom.border_sites[3] == 7);
}

TEST_CASE("build_hubbard_block_hamiltonian vacuum energy is zero") {
  auto geom = brg::block_2d_2x2();
  double t = 1.0, U = -4.0, mu = 0.0;

  Expression H = brg::build_hubbard_block_hamiltonian(geom, t, U, mu);

  // Vacuum sector: N=0, Sz=0
  Basis vacuum = Basis::with_fixed_particle_number_and_spin(geom.num_sites, 0, 0);
  arma::cx_mat mat = compute_matrix_elements<arma::cx_mat>(vacuum, H);

  CHECK(mat.n_rows == 1);
  CHECK(mat.n_cols == 1);
  CHECK(std::abs(mat(0, 0)) < 1e-12);
}

TEST_CASE("build_2d_block_hamiltonian matches generic builder") {
  double t = 1.5, U = -3.0, mu = 0.2;

  Expression H_generic = brg::build_hubbard_block_hamiltonian(brg::block_2d_2x2(), t, U, mu);
  Expression H_2d = brg::build_2d_block_hamiltonian(t, U, mu);

  // Compare in single-particle sector
  Basis basis = Basis::with_fixed_particle_number_and_spin(4, 1, 1);
  arma::cx_mat mat_generic = compute_matrix_elements<arma::cx_mat>(basis, H_generic);
  arma::cx_mat mat_2d = compute_matrix_elements<arma::cx_mat>(basis, H_2d);

  for (size_t i = 0; i < mat_generic.n_rows; ++i) {
    for (size_t j = 0; j < mat_generic.n_cols; ++j) {
      CHECK(std::abs(mat_generic(i, j) - mat_2d(i, j)) < 1e-12);
    }
  }
}

TEST_CASE("build_3d_block_hamiltonian matches generic builder") {
  double t = 0.8, U = -5.0, mu = 0.1;

  Expression H_generic = brg::build_hubbard_block_hamiltonian(brg::block_3d_2x2x2(), t, U, mu);
  Expression H_3d = brg::build_3d_block_hamiltonian(t, U, mu);

  // Compare in single-particle sector
  Basis basis = Basis::with_fixed_particle_number_and_spin(8, 1, 1);
  arma::cx_mat mat_generic = compute_matrix_elements<arma::cx_mat>(basis, H_generic);
  arma::cx_mat mat_3d = compute_matrix_elements<arma::cx_mat>(basis, H_3d);

  for (size_t i = 0; i < mat_generic.n_rows; ++i) {
    for (size_t j = 0; j < mat_generic.n_cols; ++j) {
      CHECK(std::abs(mat_generic(i, j) - mat_3d(i, j)) < 1e-12);
    }
  }
}

TEST_CASE("Hubbard block Hamiltonian is Hermitian") {
  auto geom = brg::block_2d_2x2();
  double t = 1.0, U = -4.0, mu = 0.5;

  Expression H = brg::build_hubbard_block_hamiltonian(geom, t, U, mu);

  // Test in N=2, Sz=0 sector
  Basis basis = Basis::with_fixed_particle_number_and_spin(geom.num_sites, 2, 0);
  arma::cx_mat mat = compute_matrix_elements<arma::cx_mat>(basis, H);

  // Check Hermiticity: H = H†
  arma::cx_mat mat_dag = mat.t();
  for (size_t i = 0; i < mat.n_rows; ++i) {
    for (size_t j = 0; j < mat.n_cols; ++j) {
      CHECK(std::abs(mat(i, j) - mat_dag(i, j)) < 1e-12);
    }
  }
}

TEST_CASE("Chemical potential shifts energies correctly") {
  auto geom = brg::block_2d_2x2();
  double t = 1.0, U = -4.0;

  Expression H0 = brg::build_hubbard_block_hamiltonian(geom, t, U, 0.0);
  double mu = 1.5;
  Expression H_mu = brg::build_hubbard_block_hamiltonian(geom, t, U, mu);

  // In N=2 sector, energy should shift by -2*mu
  Basis basis = Basis::with_fixed_particle_number_and_spin(geom.num_sites, 2, 0);
  arma::cx_mat mat0 = compute_matrix_elements<arma::cx_mat>(basis, H0);
  arma::cx_mat mat_mu = compute_matrix_elements<arma::cx_mat>(basis, H_mu);

  arma::vec evals0, evals_mu;
  arma::eig_sym(evals0, mat0);
  arma::eig_sym(evals_mu, mat_mu);

  // All eigenvalues should shift by -2*mu
  double expected_shift = -2.0 * mu;
  for (size_t i = 0; i < evals0.n_elem; ++i) {
    CHECK(evals_mu(i) - evals0(i) == Approx(expected_shift).margin(1e-12));
  }
}
