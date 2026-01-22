#include <armadillo>
#include <catch2/catch.hpp>
#include <cmath>
#include <complex>
#include <numbers>

#include "numerics/hubbard_relative3_operators.h"
#include "numerics/linear_operator_utils.h"
#include "utils/index.h"

namespace test_hubbard_relative3 {

TEST_CASE("hubbard_relative3_exchange_coords_1d") {
  std::vector<size_t> size = {4};
  std::vector<size_t> coords = {1, 3};

  const auto swapped = exchange_coords(coords, size);
  CHECK(swapped.size() == coords.size());
  CHECK(swapped[0] == 3);
  CHECK(swapped[1] == 2);
}

TEST_CASE("hubbard_relative3_exchange_coords_2d") {
  std::vector<size_t> size = {4, 5};
  std::vector<size_t> coords = {1, 2, 3, 4};

  const auto swapped = exchange_coords(coords, size);
  CHECK(swapped.size() == coords.size());
  CHECK(swapped[0] == 3);
  CHECK(swapped[1] == 3);
  CHECK(swapped[2] == 2);
  CHECK(swapped[3] == 2);
}

TEST_CASE("hubbard_relative3_exchange_phase_1d") {
  std::vector<size_t> size = {4};
  std::vector<size_t> K_canon = {1};
  std::vector<size_t> coords = {2, 1};

  const std::complex<double> phase = exchange_phase(coords, size, K_canon);
  CHECK(std::abs(phase.real() + 1.0) < 1e-12);
  CHECK(std::abs(phase.imag()) < 1e-12);
}

TEST_CASE("hubbard_relative3_project_antisymmetric_properties") {
  std::vector<size_t> size = {4};
  std::vector<size_t> K_canon = {1};
  std::vector<int64_t> total_momentum = {1};
  Index index({size[0], size[0]});
  HubbardRelative3 hamiltonian_rel(size, total_momentum, 1.0, 1.0);

  arma::cx_vec v(index.size(), arma::fill::zeros);
  const auto coords = std::vector<size_t>{1, 2};
  const auto partner = exchange_coords(coords, size);
  const size_t i = index(coords);
  const size_t j = index(partner);

  v(i) = {1.0, 0.5};
  v(j) = {-0.25, 0.75};

  const auto w = hamiltonian_rel.project_antisymmetric(v);
  const std::complex<double> phase = exchange_phase(coords, size, K_canon);

  CHECK(std::abs(w(i) - 0.5 * (v(i) - phase * v(j))) < 1e-12);
  CHECK(std::abs(w(i) + phase * w(j)) < 1e-12);

  const auto w2 = hamiltonian_rel.project_antisymmetric(w);
  CHECK(arma::norm(w2 - w) < 1e-12);
}

TEST_CASE("hubbard_relative3_interaction_dimension_1d") {
  std::vector<size_t> size = {4};
  HubbardRelative3Interaction op(size);

  // Dimension should be L^2 = 4^2 = 16 (r2 in [0,3], r3 in [0,3])
  CHECK(op.dimension() == 16);
}

TEST_CASE("hubbard_relative3_interaction_dimension_2d") {
  std::vector<size_t> size = {3, 4};
  HubbardRelative3Interaction op(size);

  // Dimension should be (3*4)^2 = 144
  CHECK(op.dimension() == 144);
}

TEST_CASE("hubbard_relative3_interaction_is_diagonal") {
  std::vector<size_t> size = {4};
  HubbardRelative3Interaction op(size);

  // Check that V is diagonal by applying to basis vectors
  for (size_t i = 0; i < op.dimension(); ++i) {
    arma::cx_vec e_i(op.dimension(), arma::fill::zeros);
    e_i(i) = 1.0;
    arma::cx_vec result = op.apply(e_i);

    // Only the i-th component should be non-zero
    for (size_t j = 0; j < op.dimension(); ++j) {
      if (i != j) {
        CHECK(std::abs(result(j)) < 1e-14);
      }
    }
  }
}

TEST_CASE("hubbard_relative3_interaction_overlap_counts_1d") {
  std::vector<size_t> size = {4};
  HubbardRelative3Interaction op(size);

  // In 1D with L=4, the index maps (r2, r3) to orbital = r2 + 4*r3
  // r3=0 and r3=r2 gives overlap

  arma::cx_vec v(op.dimension(), arma::fill::ones);
  arma::cx_vec w = op.apply(v);

  // At (r2=0, r3=0): r3==0 AND r3==r2, so overlap count = 2
  CHECK(std::abs(w(0) - 2.0) < 1e-14);

  // At (r2=1, r3=0): r3==0 but r3!=r2, so overlap count = 1
  CHECK(std::abs(w(1) - 1.0) < 1e-14);

  // At (r2=1, r3=1): r3!=0 but r3==r2, so overlap count = 1
  // orbital = 1 + 4*1 = 5
  CHECK(std::abs(w(5) - 1.0) < 1e-14);

  // At (r2=2, r3=3): r3!=0 and r3!=r2, so overlap count = 0
  // orbital = 2 + 4*3 = 14
  CHECK(std::abs(w(14)) < 1e-14);
}

TEST_CASE("hubbard_relative3_kinetic_dimension_matches_interaction") {
  std::vector<size_t> size = {4};
  std::vector<int64_t> momentum = {0};

  HubbardRelative3Interaction interaction(size);
  HubbardRelative3Kinetic kinetic(size, momentum);

  CHECK(kinetic.dimension() == interaction.dimension());
}

TEST_CASE("hubbard_relative3_kinetic_zero_momentum_is_hermitian") {
  std::vector<size_t> size = {4};
  std::vector<int64_t> momentum = {0};
  HubbardRelative3Kinetic op(size, momentum);

  // Build the full matrix
  arma::cx_mat K(op.dimension(), op.dimension());
  for (size_t j = 0; j < op.dimension(); ++j) {
    arma::cx_vec e_j(op.dimension(), arma::fill::zeros);
    e_j(j) = 1.0;
    K.col(j) = op.apply(e_j);
  }

  // Check Hermiticity
  arma::cx_mat diff = K - K.t();
  CHECK(arma::norm(diff, "fro") < 1e-12);
}

TEST_CASE("hubbard_relative3_kinetic_nonzero_momentum_has_correct_phases") {
  std::vector<size_t> size = {4};
  std::vector<int64_t> momentum = {1};
  HubbardRelative3Kinetic op(size, momentum);

  // Build full matrix
  arma::cx_mat K(op.dimension(), op.dimension());
  for (size_t j = 0; j < op.dimension(); ++j) {
    arma::cx_vec e_j(op.dimension(), arma::fill::zeros);
    e_j(j) = 1.0;
    K.col(j) = op.apply(e_j);
  }

  // Matrix should still be Hermitian for any momentum
  arma::cx_mat diff = K - K.t();
  CHECK(arma::norm(diff, "fro") < 1e-12);

  // Check that some off-diagonal elements are complex (non-zero imaginary part)
  // when momentum is non-zero
  bool has_complex = false;
  for (size_t i = 0; i < op.dimension(); ++i) {
    for (size_t j = 0; j < op.dimension(); ++j) {
      if (std::abs(K(i, j).imag()) > 1e-10) {
        has_complex = true;
        break;
      }
    }
    if (has_complex) break;
  }
  CHECK(has_complex);
}

TEST_CASE("hubbard_relative3_kinetic_reference_phase_matches_expected_1d") {
  const size_t L = 4;
  std::vector<size_t> size = {L};
  std::vector<int64_t> momentum = {1};
  HubbardRelative3Kinetic op(size, momentum);

  Index index({L, L});
  const size_t dest = index({0, 0});

  arma::cx_vec v(op.dimension(), arma::fill::zeros);
  v(index({1, 1})) = 1.0;
  arma::cx_vec w = op.apply(v);
  std::complex<double> expected =
      std::exp(std::complex<double>(0.0, 2.0 * std::numbers::pi / static_cast<double>(L)));
  CHECK(std::abs(w(dest) - expected) < 1e-12);

  v.zeros();
  v(index({L - 1, L - 1})) = 1.0;
  w = op.apply(v);
  expected = std::exp(std::complex<double>(0.0, -2.0 * std::numbers::pi / static_cast<double>(L)));
  CHECK(std::abs(w(dest) - expected) < 1e-12);
}

TEST_CASE("hubbard_relative3_kinetic_preserves_translation_symmetry") {
  // For zero total momentum, kinetic should have real spectrum
  std::vector<size_t> size = {4};
  std::vector<int64_t> momentum = {0};
  HubbardRelative3Kinetic op(size, momentum);

  arma::cx_mat K(op.dimension(), op.dimension());
  for (size_t j = 0; j < op.dimension(); ++j) {
    arma::cx_vec e_j(op.dimension(), arma::fill::zeros);
    e_j(j) = 1.0;
    K.col(j) = op.apply(e_j);
  }

  arma::cx_vec eigvals;
  arma::eig_gen(eigvals, K);

  // All eigenvalues should be real for K=0
  for (size_t i = 0; i < eigvals.n_elem; ++i) {
    CHECK(std::abs(eigvals(i).imag()) < 1e-10);
  }
}

TEST_CASE("hubbard_relative3_full_operator_combines_kinetic_and_interaction") {
  std::vector<size_t> size = {4};
  std::vector<int64_t> momentum = {0};
  double t = 1.0;
  double U = 4.0;

  HubbardRelative3 full(size, momentum, t, U);
  HubbardRelative3Kinetic kinetic(size, momentum);
  HubbardRelative3Interaction interaction(size);

  arma::cx_vec v = arma::randu<arma::cx_vec>(full.dimension());

  arma::cx_vec full_result = full.apply(v);
  arma::cx_vec manual_result = -t * kinetic.apply(v) + U * interaction.apply(v);

  CHECK(arma::norm(full_result - manual_result) < 1e-12);
}

TEST_CASE("hubbard_relative3_full_operator_is_hermitian") {
  std::vector<size_t> size = {3};
  std::vector<int64_t> momentum = {1};
  double t = 1.0;
  double U = 2.0;

  HubbardRelative3 op(size, momentum, t, U);

  arma::cx_mat H(op.dimension(), op.dimension());
  for (size_t j = 0; j < op.dimension(); ++j) {
    arma::cx_vec e_j(op.dimension(), arma::fill::zeros);
    e_j(j) = 1.0;
    H.col(j) = op.apply(e_j);
  }

  arma::cx_mat diff = H - H.t();
  CHECK(arma::norm(diff, "fro") < 1e-12);
}

TEST_CASE("hubbard_relative3_2d_system") {
  std::vector<size_t> size = {3, 3};
  std::vector<int64_t> momentum = {0, 0};

  HubbardRelative3 op(size, momentum, 1.0, 2.0);

  // Dimension should be (3*3)^2 = 81
  CHECK(op.dimension() == 81);

  // Full operator should be Hermitian
  arma::cx_mat H(op.dimension(), op.dimension());
  for (size_t j = 0; j < op.dimension(); ++j) {
    arma::cx_vec e_j(op.dimension(), arma::fill::zeros);
    e_j(j) = 1.0;
    H.col(j) = op.apply(e_j);
  }

  arma::cx_mat diff = H - H.t();
  CHECK(arma::norm(diff, "fro") < 1e-12);
}

TEST_CASE("hubbard_relative3_momentum_canonicalization") {
  std::vector<size_t> size = {4};

  // Momentum K=5 should be canonicalized to K=1 (mod 4)
  std::vector<int64_t> momentum_5 = {5};
  std::vector<int64_t> momentum_1 = {1};

  HubbardRelative3Kinetic op5(size, momentum_5);
  HubbardRelative3Kinetic op1(size, momentum_1);

  // Both operators should produce identical results
  arma::cx_vec v = arma::randu<arma::cx_vec>(op5.dimension());
  arma::cx_vec result5 = op5.apply(v);
  arma::cx_vec result1 = op1.apply(v);

  CHECK(arma::norm(result5 - result1) < 1e-12);
}

TEST_CASE("hubbard_relative3_negative_momentum_canonicalization") {
  std::vector<size_t> size = {4};

  // Momentum K=-1 should be canonicalized to K=3 (mod 4)
  std::vector<int64_t> momentum_neg = {-1};
  std::vector<int64_t> momentum_pos = {3};

  HubbardRelative3Kinetic op_neg(size, momentum_neg);
  HubbardRelative3Kinetic op_pos(size, momentum_pos);

  arma::cx_vec v = arma::randu<arma::cx_vec>(op_neg.dimension());
  arma::cx_vec result_neg = op_neg.apply(v);
  arma::cx_vec result_pos = op_pos.apply(v);

  CHECK(arma::norm(result_neg - result_pos) < 1e-12);
}

TEST_CASE("hubbard_relative3_kinetic_hop_count_per_orbital") {
  // In 1D with K=0, each orbital should have 6 hopping terms:
  // - 2 from particle 2 (left/right in r2)
  // - 2 from particle 3 (left/right in r3)
  // - 2 from particle 1 (simultaneous shift of r2,r3)

  std::vector<size_t> size = {6};
  std::vector<int64_t> momentum = {0};
  HubbardRelative3Kinetic op(size, momentum);

  arma::cx_mat K(op.dimension(), op.dimension());
  for (size_t j = 0; j < op.dimension(); ++j) {
    arma::cx_vec e_j(op.dimension(), arma::fill::zeros);
    e_j(j) = 1.0;
    K.col(j) = op.apply(e_j);
  }

  // Count non-zero off-diagonal elements for each row
  for (size_t i = 0; i < op.dimension(); ++i) {
    size_t count = 0;
    for (size_t j = 0; j < op.dimension(); ++j) {
      if (i != j && std::abs(K(i, j)) > 1e-14) {
        ++count;
      }
    }
    // Each orbital connects to exactly 6 neighbors (for generic positions)
    CHECK(count == 6);
  }
}

}  // namespace test_hubbard_relative3
