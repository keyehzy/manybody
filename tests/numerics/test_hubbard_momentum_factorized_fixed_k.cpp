#include <armadillo>
#include <catch2/catch.hpp>
#include <cmath>
#include <numbers>
#include <vector>

#include "algebra/fermion/basis.h"
#include "algebra/matrix_elements.h"
#include "algebra/model/hubbard_model_momentum.h"
#include "numerics/hubbard_momentum_factorized_fixed_k.h"

namespace {
constexpr double kTolerance = 1e-10;
}

TEST_CASE("hubbard_momentum_factorized_fixed_k_construction_1d") {
  const std::vector<size_t> size = {4};
  const size_t sites = 4;
  const size_t particles = 2;
  const int spin = 0;
  const Index index(size);
  const Index::container_type target_momentum = {0};

  Basis basis = Basis::with_fixed_particle_number_spin_momentum(sites, particles, spin, index,
                                                                target_momentum);
  HubbardMomentumFactorizedFixedK H(basis, size, 1.0, 2.0, target_momentum);

  CHECK(H.dimension() == basis.set.size());
}

TEST_CASE("hubbard_momentum_factorized_fixed_k_kinetic_diagonal_1d") {
  const std::vector<size_t> size = {4};
  const size_t sites = 4;
  const size_t particles = 2;
  const int spin = 0;
  const Index index(size);
  const Index::container_type target_momentum = {0};

  Basis basis = Basis::with_fixed_particle_number_spin_momentum(sites, particles, spin, index,
                                                                target_momentum);
  HubbardMomentumFactorizedFixedK H(basis, size, 1.0, 0.0, target_momentum);  // U=0, only kinetic

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

TEST_CASE("hubbard_momentum_factorized_fixed_k_kinetic_matches_dense_1d") {
  const std::vector<size_t> size = {4};
  const size_t sites = 4;
  const size_t particles = 2;
  const int spin = 0;
  const double t = 1.5;
  const Index index(size);
  const Index::container_type target_momentum = {1};

  Basis basis = Basis::with_fixed_particle_number_spin_momentum(sites, particles, spin, index,
                                                                target_momentum);

  // Factorized version
  HubbardMomentumFactorizedFixedK H_factorized(basis, size, t, 0.0, target_momentum);

  // Dense version using HubbardModelMomentum
  HubbardModelMomentum model(t, 0.0, size);
  arma::cx_mat H_dense = compute_matrix_elements<arma::cx_mat>(basis, model.kinetic());

  // Compare application on random vectors
  arma::cx_vec v = arma::randn<arma::cx_vec>(basis.set.size());
  arma::cx_vec w_factorized = H_factorized.apply(v);
  arma::cx_vec w_dense = H_dense * v;

  CHECK(arma::norm(w_factorized - w_dense) < kTolerance * arma::norm(w_dense));
}

TEST_CASE("hubbard_momentum_factorized_fixed_k_interaction_matches_dense_1d_small") {
  const std::vector<size_t> size = {2};
  const size_t sites = 2;
  const size_t particles = 2;
  const int spin = 0;
  const double U = 3.0;
  const Index index(size);
  const Index::container_type target_momentum = {0};

  Basis basis = Basis::with_fixed_particle_number_spin_momentum(sites, particles, spin, index,
                                                                target_momentum);

  // Factorized version (t=0 to isolate interaction)
  HubbardMomentumFactorizedFixedK H_factorized(basis, size, 0.0, U, target_momentum);

  // Dense version
  HubbardModelMomentum model(0.0, U, size);
  arma::cx_mat H_dense = compute_matrix_elements<arma::cx_mat>(basis, model.interaction());

  // Compare application on random vectors
  arma::cx_vec v = arma::randn<arma::cx_vec>(basis.set.size());
  arma::cx_vec w_factorized = H_factorized.apply(v);
  arma::cx_vec w_dense = H_dense * v;

  CHECK(arma::norm(w_factorized - w_dense) < kTolerance * (1.0 + arma::norm(w_dense)));
}

TEST_CASE("hubbard_momentum_factorized_fixed_k_full_hamiltonian_matches_dense_1d") {
  const std::vector<size_t> size = {4};
  const size_t sites = 4;
  const size_t particles = 2;
  const int spin = 0;
  const double t = 1.0;
  const double U = 4.0;
  const Index index(size);
  const Index::container_type target_momentum = {2};

  Basis basis = Basis::with_fixed_particle_number_spin_momentum(sites, particles, spin, index,
                                                                target_momentum);

  // Factorized version
  HubbardMomentumFactorizedFixedK H_factorized(basis, size, t, U, target_momentum);

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

TEST_CASE("hubbard_momentum_factorized_fixed_k_full_hamiltonian_matches_dense_1d_larger") {
  const std::vector<size_t> size = {6};
  const size_t sites = 6;
  const size_t particles = 3;
  const int spin = 1;  // 2 up, 1 down
  const double t = 1.0;
  const double U = -2.0;  // Attractive
  const Index index(size);
  const Index::container_type target_momentum = {3};

  Basis basis = Basis::with_fixed_particle_number_spin_momentum(sites, particles, spin, index,
                                                                target_momentum);

  // Factorized version
  HubbardMomentumFactorizedFixedK H_factorized(basis, size, t, U, target_momentum);

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

TEST_CASE("hubbard_momentum_factorized_fixed_k_2d") {
  const std::vector<size_t> size = {2, 2};
  const size_t sites = 4;
  const size_t particles = 2;
  const int spin = 0;
  const double t = 1.0;
  const double U = 2.0;
  const Index index(size);
  const Index::container_type target_momentum = {1, 0};

  Basis basis = Basis::with_fixed_particle_number_spin_momentum(sites, particles, spin, index,
                                                                target_momentum);

  // Factorized version
  HubbardMomentumFactorizedFixedK H_factorized(basis, size, t, U, target_momentum);

  // Dense version
  HubbardModelMomentum model(t, U, size);
  arma::cx_mat H_dense = compute_matrix_elements<arma::cx_mat>(basis, model.hamiltonian());

  // Compare
  arma::cx_vec v = arma::randn<arma::cx_vec>(basis.set.size());
  arma::cx_vec w_factorized = H_factorized.apply(v);
  arma::cx_vec w_dense = H_dense * v;

  CHECK(arma::norm(w_factorized - w_dense) < kTolerance * (1.0 + arma::norm(w_dense)));
}

TEST_CASE("hubbard_momentum_factorized_fixed_k_2d_nonzero_momentum") {
  const std::vector<size_t> size = {3, 3};
  const size_t sites = 9;
  const size_t particles = 2;
  const int spin = 0;
  const double t = 1.0;
  const double U = 4.0;
  const Index index(size);
  const Index::container_type target_momentum = {1, 2};

  Basis basis = Basis::with_fixed_particle_number_spin_momentum(sites, particles, spin, index,
                                                                target_momentum);

  // Factorized version
  HubbardMomentumFactorizedFixedK H_factorized(basis, size, t, U, target_momentum);

  // Dense version
  HubbardModelMomentum model(t, U, size);
  arma::cx_mat H_dense = compute_matrix_elements<arma::cx_mat>(basis, model.hamiltonian());

  // Compare on multiple trials
  for (int trial = 0; trial < 3; ++trial) {
    arma::cx_vec v = arma::randn<arma::cx_vec>(basis.set.size());
    arma::cx_vec w_factorized = H_factorized.apply(v);
    arma::cx_vec w_dense = H_dense * v;

    const double err = arma::norm(w_factorized - w_dense);
    const double scale = 1.0 + arma::norm(w_dense);
    CHECK(err < kTolerance * scale);
  }
}

TEST_CASE("hubbard_momentum_factorized_fixed_k_hermitian") {
  const std::vector<size_t> size = {4};
  const size_t sites = 4;
  const size_t particles = 2;
  const int spin = 0;
  const double t = 1.0;
  const double U = 3.0;
  const Index index(size);
  const Index::container_type target_momentum = {1};

  Basis basis = Basis::with_fixed_particle_number_spin_momentum(sites, particles, spin, index,
                                                                target_momentum);
  HubbardMomentumFactorizedFixedK H(basis, size, t, U, target_momentum);

  // Check ⟨x|H|y⟩ = ⟨y|H|x⟩* for random vectors
  arma::cx_vec x = arma::randn<arma::cx_vec>(basis.set.size());
  arma::cx_vec y = arma::randn<arma::cx_vec>(basis.set.size());

  arma::cx_double x_H_y = arma::cdot(x, H.apply(y));
  arma::cx_double y_H_x = arma::cdot(y, H.apply(x));

  CHECK(std::abs(x_H_y - std::conj(y_H_x)) < kTolerance * std::abs(x_H_y));
}

TEST_CASE("hubbard_momentum_factorized_fixed_k_sector_bases_correct_size") {
  const std::vector<size_t> size = {4};
  const size_t sites = 4;
  const size_t particles = 2;
  const int spin = 0;
  const Index index(size);
  const Index::container_type target_momentum = {0};

  Basis basis = Basis::with_fixed_particle_number_spin_momentum(sites, particles, spin, index,
                                                                target_momentum);
  HubbardMomentumFactorizedFixedK H(basis, size, 1.0, 1.0, target_momentum);

  const auto& sector_bases = H.sector_bases();
  CHECK(sector_bases.size() == sites);

  // sector_bases[0] should be the same as the input basis (sector K)
  CHECK(sector_bases[0].set.size() == basis.set.size());
}

TEST_CASE("hubbard_momentum_factorized_fixed_k_density_operators_rectangular") {
  const std::vector<size_t> size = {4};
  const size_t sites = 4;
  const size_t particles = 2;
  const int spin = 0;
  const Index index(size);
  const Index::container_type target_momentum = {1};

  Basis basis = Basis::with_fixed_particle_number_spin_momentum(sites, particles, spin, index,
                                                                target_momentum);
  HubbardMomentumFactorizedFixedK H(basis, size, 1.0, 1.0, target_momentum);

  const auto& rho_up = H.rho_up();
  const auto& rho_down = H.rho_down();
  const auto& sector_bases = H.sector_bases();

  for (size_t q = 0; q < sites; ++q) {
    // rho_up_[q] maps K-q → K, so shape is dim(K) × dim(K-q)
    // K - q corresponds to sector_bases[minus_q]
    size_t minus_q = (sites - q) % sites;
    CHECK(rho_up[q].n_rows == basis.set.size());
    CHECK(rho_up[q].n_cols == sector_bases[minus_q].set.size());

    // rho_down_[q] maps K → K+q, so shape is dim(K+q) × dim(K)
    CHECK(rho_down[q].n_rows == sector_bases[q].set.size());
    CHECK(rho_down[q].n_cols == basis.set.size());
  }
}

TEST_CASE("hubbard_momentum_factorized_fixed_k_all_momentum_sectors_1d") {
  // Test that the factorized version matches dense for ALL momentum sectors
  const std::vector<size_t> size = {4};
  const size_t sites = 4;
  const size_t particles = 2;
  const int spin = 0;
  const double t = 1.0;
  const double U = 4.0;
  const Index index(size);

  HubbardModelMomentum model(t, U, size);

  for (size_t k = 0; k < sites; ++k) {
    const Index::container_type target_momentum = {k};

    Basis basis = Basis::with_fixed_particle_number_spin_momentum(sites, particles, spin, index,
                                                                  target_momentum);

    if (basis.set.empty()) {
      continue;  // Skip empty sectors
    }

    HubbardMomentumFactorizedFixedK H_factorized(basis, size, t, U, target_momentum);
    arma::cx_mat H_dense = compute_matrix_elements<arma::cx_mat>(basis, model.hamiltonian());

    arma::cx_vec v = arma::randn<arma::cx_vec>(basis.set.size());
    arma::cx_vec w_factorized = H_factorized.apply(v);
    arma::cx_vec w_dense = H_dense * v;

    const double err = arma::norm(w_factorized - w_dense);
    const double scale = 1.0 + arma::norm(w_dense);
    CHECK(err < kTolerance * scale);
  }
}

TEST_CASE("hubbard_momentum_factorized_fixed_k_all_momentum_sectors_2d") {
  // Test that the factorized version matches dense for ALL momentum sectors in 2D
  const std::vector<size_t> size = {2, 2};
  const size_t sites = 4;
  const size_t particles = 2;
  const int spin = 0;
  const double t = 1.0;
  const double U = 4.0;
  const Index index(size);

  HubbardModelMomentum model(t, U, size);

  for (size_t kx = 0; kx < size[0]; ++kx) {
    for (size_t ky = 0; ky < size[1]; ++ky) {
      const Index::container_type target_momentum = {kx, ky};

      Basis basis = Basis::with_fixed_particle_number_spin_momentum(sites, particles, spin, index,
                                                                    target_momentum);

      if (basis.set.empty()) {
        continue;  // Skip empty sectors
      }

      HubbardMomentumFactorizedFixedK H_factorized(basis, size, t, U, target_momentum);
      arma::cx_mat H_dense = compute_matrix_elements<arma::cx_mat>(basis, model.hamiltonian());

      arma::cx_vec v = arma::randn<arma::cx_vec>(basis.set.size());
      arma::cx_vec w_factorized = H_factorized.apply(v);
      arma::cx_vec w_dense = H_dense * v;

      const double err = arma::norm(w_factorized - w_dense);
      const double scale = 1.0 + arma::norm(w_dense);
      CHECK(err < kTolerance * scale);
    }
  }
}
