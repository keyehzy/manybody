#include <armadillo>
#include <cmath>

#include "algebra/basis.h"
#include "algebra/matrix_elements.h"
#include "algebra/model/hubbard_model.h"
#include "algorithms/schriffer_wolff.h"
#include "framework.h"

TEST(schriffer_wolff_cluster_by_largest_gap) {
  arma::vec vals = {0.0, 0.5, 0.8, 2.0};
  const auto [split, gap] = cluster_by_largest_gap(vals);
  EXPECT_EQ(split, static_cast<size_t>(3));
  EXPECT_TRUE(std::abs(gap - 1.2) < 1e-12);
}

TEST(schriffer_wolff_zero_kinetic_returns_zero_generator) {
  const size_t lattice_size = 2;
  const size_t particles = 2;
  HubbardModel hubbard(0.0, 8.0, lattice_size);
  Basis basis = Basis::with_fixed_particle_number(lattice_size, particles);

  Expression generator = schriffer_wolff(hubbard.kinetic(), hubbard.interaction(), basis, 10);
  arma::cx_mat A = compute_matrix_elements<arma::cx_mat>(basis, generator);
  const double max_val = arma::abs(A).max();
  EXPECT_TRUE(max_val < 1e-10);
}

TEST(schriffer_wolff_generator_is_antihermitian) {
  const size_t lattice_size = 2;
  const size_t particles = 2;
  HubbardModel hubbard(1.0, 8.0, lattice_size);
  Basis basis = Basis::with_fixed_particle_number(lattice_size, particles);

  Expression generator = schriffer_wolff(hubbard.kinetic(), hubbard.interaction(), basis, 50);
  arma::cx_mat A = compute_matrix_elements<arma::cx_mat>(basis, generator);
  arma::cx_mat anti = A + A.st();
  const double max_val = arma::abs(anti).max();
  EXPECT_TRUE(max_val < 1e-6);
}
