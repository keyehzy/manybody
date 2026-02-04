#include <armadillo>
#include <catch2/catch.hpp>
#include <cmath>
#include <numbers>
#include <vector>

#include "algebra/basis.h"
#include "algebra/matrix_elements.h"
#include "algebra/model/hubbard_model_momentum.h"
#include "numerics/hubbard_momentum_factorized.h"

namespace {
constexpr double kTolerance = 1e-10;
}

TEST_CASE("hubbard_momentum_factorized_construction_1d") {
  const std::vector<size_t> size = {4};
  const size_t sites = 4;
  const size_t particles = 2;
  const int spin = 0;

  Basis basis = Basis::with_fixed_particle_number_and_spin(sites, particles, spin);
  HubbardMomentumFactorized H(basis, size, 1.0, 2.0);

  CHECK(H.dimension() == basis.set.size());
}

TEST_CASE("hubbard_momentum_factorized_kinetic_diagonal_1d") {
  const std::vector<size_t> size = {4};
  const size_t sites = 4;
  const size_t particles = 2;
  const int spin = 0;

  Basis basis = Basis::with_fixed_particle_number_and_spin(sites, particles, spin);
  HubbardMomentumFactorized H(basis, size, 1.0, 0.0);  // U=0, only kinetic

  // Apply to a basis vector and check the result
  arma::cx_vec v(basis.set.size(), arma::fill::zeros);
  v(0) = 1.0;  // First basis state

  arma::cx_vec w = H.apply(v);

  // The result should be the kinetic energy times v
  // Check that it's diagonal in the basis
  for (size_t i = 0; i < basis.set.size(); ++i) {
    if (i == 0) {
      CHECK(std::abs(w(i) - H.kinetic_diagonal()(0)) < kTolerance);
    } else {
      CHECK(std::abs(w(i)) < kTolerance);
    }
  }
}

TEST_CASE("hubbard_momentum_factorized_kinetic_matches_dense_1d") {
  const std::vector<size_t> size = {4};
  const size_t sites = 4;
  const size_t particles = 2;
  const int spin = 0;
  const double t = 1.5;

  Basis basis = Basis::with_fixed_particle_number_and_spin(sites, particles, spin);

  // Factorized version
  HubbardMomentumFactorized H_factorized(basis, size, t, 0.0);

  // Dense version using HubbardModelMomentum
  HubbardModelMomentum model(t, 0.0, size);
  arma::cx_mat H_dense = compute_matrix_elements<arma::cx_mat>(basis, model.kinetic());

  // Compare application on random vectors
  arma::cx_vec v = arma::randn<arma::cx_vec>(basis.set.size());
  arma::cx_vec w_factorized = H_factorized.apply(v);
  arma::cx_vec w_dense = H_dense * v;

  CHECK(arma::norm(w_factorized - w_dense) < kTolerance * arma::norm(w_dense));
}

TEST_CASE("hubbard_momentum_factorized_interaction_matches_dense_1d_small") {
  const std::vector<size_t> size = {2};
  const size_t sites = 2;
  const size_t particles = 2;
  const int spin = 0;
  const double U = 3.0;

  Basis basis = Basis::with_fixed_particle_number_and_spin(sites, particles, spin);

  // Factorized version (t=0 to isolate interaction)
  HubbardMomentumFactorized H_factorized(basis, size, 0.0, U);

  // Dense version
  HubbardModelMomentum model(0.0, U, size);
  arma::cx_mat H_dense = compute_matrix_elements<arma::cx_mat>(basis, model.interaction());

  // Compare application on random vectors
  arma::cx_vec v = arma::randn<arma::cx_vec>(basis.set.size());
  arma::cx_vec w_factorized = H_factorized.apply(v);
  arma::cx_vec w_dense = H_dense * v;

  CHECK(arma::norm(w_factorized - w_dense) < kTolerance * (1.0 + arma::norm(w_dense)));
}

TEST_CASE("hubbard_momentum_factorized_full_hamiltonian_matches_dense_1d") {
  const std::vector<size_t> size = {4};
  const size_t sites = 4;
  const size_t particles = 2;
  const int spin = 0;
  const double t = 1.0;
  const double U = 4.0;

  Basis basis = Basis::with_fixed_particle_number_and_spin(sites, particles, spin);

  // Factorized version
  HubbardMomentumFactorized H_factorized(basis, size, t, U);

  // Dense version
  HubbardModelMomentum model(t, U, size);
  arma::cx_mat H_dense = compute_matrix_elements<arma::cx_mat>(basis, model.hamiltonian());

  // Compare application on multiple random vectors
  for (int trial = 0; trial < 5; ++trial) {
    arma::cx_vec v = arma::randn<arma::cx_vec>(basis.set.size());
    arma::cx_vec w_factorized = H_factorized.apply(v);
    arma::cx_vec w_dense = H_dense * v;

    const double err = arma::norm(w_factorized - w_dense);
    const double scale = 1.0 + arma::norm(w_dense);
    CHECK(err < kTolerance * scale);
  }
}

TEST_CASE("hubbard_momentum_factorized_full_hamiltonian_matches_dense_1d_larger") {
  const std::vector<size_t> size = {6};
  const size_t sites = 6;
  const size_t particles = 3;
  const int spin = 1;  // 2 up, 1 down
  const double t = 1.0;
  const double U = -2.0;  // Attractive

  Basis basis = Basis::with_fixed_particle_number_and_spin(sites, particles, spin);

  // Factorized version
  HubbardMomentumFactorized H_factorized(basis, size, t, U);

  // Dense version
  HubbardModelMomentum model(t, U, size);
  arma::cx_mat H_dense = compute_matrix_elements<arma::cx_mat>(basis, model.hamiltonian());

  // Compare on random vectors
  for (int trial = 0; trial < 3; ++trial) {
    arma::cx_vec v = arma::randn<arma::cx_vec>(basis.set.size());
    arma::cx_vec w_factorized = H_factorized.apply(v);
    arma::cx_vec w_dense = H_dense * v;

    const double err = arma::norm(w_factorized - w_dense);
    const double scale = 1.0 + arma::norm(w_dense);
    CHECK(err < kTolerance * scale);
  }
}

TEST_CASE("hubbard_momentum_factorized_2d") {
  const std::vector<size_t> size = {2, 2};
  const size_t sites = 4;
  const size_t particles = 2;
  const int spin = 0;
  const double t = 1.0;
  const double U = 2.0;

  Basis basis = Basis::with_fixed_particle_number_and_spin(sites, particles, spin);

  // Factorized version
  HubbardMomentumFactorized H_factorized(basis, size, t, U);

  // Dense version
  HubbardModelMomentum model(t, U, size);
  arma::cx_mat H_dense = compute_matrix_elements<arma::cx_mat>(basis, model.hamiltonian());

  // Compare
  arma::cx_vec v = arma::randn<arma::cx_vec>(basis.set.size());
  arma::cx_vec w_factorized = H_factorized.apply(v);
  arma::cx_vec w_dense = H_dense * v;

  CHECK(arma::norm(w_factorized - w_dense) < kTolerance * (1.0 + arma::norm(w_dense)));
}

TEST_CASE("hubbard_momentum_factorized_hermitian") {
  const std::vector<size_t> size = {4};
  const size_t sites = 4;
  const size_t particles = 2;
  const int spin = 0;
  const double t = 1.0;
  const double U = 3.0;

  Basis basis = Basis::with_fixed_particle_number_and_spin(sites, particles, spin);
  HubbardMomentumFactorized H(basis, size, t, U);

  // Check ⟨x|H|y⟩ = ⟨y|H|x⟩* for random vectors
  arma::cx_vec x = arma::randn<arma::cx_vec>(basis.set.size());
  arma::cx_vec y = arma::randn<arma::cx_vec>(basis.set.size());

  arma::cx_double x_H_y = arma::cdot(x, H.apply(y));
  arma::cx_double y_H_x = arma::cdot(y, H.apply(x));

  CHECK(std::abs(x_H_y - std::conj(y_H_x)) < kTolerance * std::abs(x_H_y));
}

TEST_CASE("hubbard_momentum_factorized_density_operators_sparse") {
  const std::vector<size_t> size = {4};
  const size_t sites = 4;
  const size_t particles = 2;
  const int spin = 0;

  Basis basis = Basis::with_fixed_particle_number_and_spin(sites, particles, spin);
  HubbardMomentumFactorized H(basis, size, 1.0, 1.0);

  // Check that density operators are sparse
  // Each column should have O(particles) non-zeros
  const auto& rho_up = H.rho_up();
  const auto& rho_down = H.rho_down();

  for (size_t q = 0; q < sites; ++q) {
    // Average non-zeros per column should be O(particles)
    const double avg_nnz_up =
        static_cast<double>(rho_up[q].n_nonzero) / static_cast<double>(basis.set.size());
    const double avg_nnz_down =
        static_cast<double>(rho_down[q].n_nonzero) / static_cast<double>(basis.set.size());

    // Should be much less than basis size (which would be dense)
    CHECK(avg_nnz_up <= static_cast<double>(particles + 1));
    CHECK(avg_nnz_down <= static_cast<double>(particles + 1));
  }
}
