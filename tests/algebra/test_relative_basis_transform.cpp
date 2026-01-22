#include <armadillo>
#include <cstddef>
#include <vector>

#include "algebra/basis.h"
#include "algebra/matrix_elements.h"
#include "algebra/model/hubbard_model_momentum.h"
#include "algebra/relative_basis_transform.h"
#include <catch2/catch.hpp>
#include "utils/index.h"

namespace {
constexpr double kRelTransformTolerance = 1e-10;
}

TEST_CASE("relative_basis_transform_unitarity") {
  const size_t lattice_size = 4;
  const std::vector<size_t> size{lattice_size};
  const size_t K = 1;

  Index index(size);
  Basis momentum_basis =
      Basis::with_fixed_particle_number_spin_momentum(lattice_size, 2, 0, index, {K});

  arma::cx_mat U = relative_position_transform(momentum_basis, index);

  const arma::cx_mat should_be_identity = U.t() * U;
  const arma::cx_mat identity = arma::eye<arma::cx_mat>(U.n_cols, U.n_cols);
  const double error = arma::norm(should_be_identity - identity, "fro");

  CHECK(error < kRelTransformTolerance);
}

TEST_CASE("relative_basis_transform_eigenvalue_preservation") {
  const size_t lattice_size = 4;
  const std::vector<size_t> size{lattice_size};
  const size_t K = 1;
  const double t = 1.0;
  const double U_hubbard = 4.0;

  Index index(size);
  Basis momentum_basis =
      Basis::with_fixed_particle_number_spin_momentum(lattice_size, 2, 0, index, {K});

  HubbardModelMomentum hubbard(t, U_hubbard, size);
  const Expression H_expr = hubbard.hamiltonian();
  arma::cx_mat H_mom = compute_matrix_elements<arma::cx_mat>(momentum_basis, H_expr);

  arma::cx_mat U = relative_position_transform(momentum_basis, index);
  const arma::cx_mat H_rel = U.t() * H_mom * U;

  arma::vec eig_mom;
  arma::eig_sym(eig_mom, arma::cx_mat(H_mom));

  arma::vec eig_rel;
  arma::eig_sym(eig_rel, arma::cx_mat(H_rel));

  eig_mom = arma::sort(eig_mom);
  eig_rel = arma::sort(eig_rel);

  const double error = arma::norm(eig_mom - eig_rel);
  CHECK(error < kRelTransformTolerance);
}

TEST_CASE("relative_basis_transform_interaction_localization") {
  const size_t lattice_size = 4;
  const std::vector<size_t> size{lattice_size};
  const size_t K = 1;
  const double t = 1.0;
  const double U_hubbard = 4.0;

  Index index(size);
  Basis momentum_basis =
      Basis::with_fixed_particle_number_spin_momentum(lattice_size, 2, 0, index, {K});

  HubbardModelMomentum hubbard(t, U_hubbard, size);
  arma::cx_mat H_interaction_mom =
      compute_matrix_elements<arma::cx_mat>(momentum_basis, hubbard.interaction());

  arma::cx_mat U = relative_position_transform(momentum_basis, index);
  arma::cx_mat H_interaction_rel = U.t() * H_interaction_mom * U;

  // The interaction should be U at (0,0) and ~0 elsewhere
  const double diag_value = std::abs(H_interaction_rel(0, 0));
  double off_diag_max = 0.0;
  for (size_t i = 0; i < H_interaction_rel.n_rows; ++i) {
    for (size_t j = 0; j < H_interaction_rel.n_cols; ++j) {
      if (i == 0 && j == 0) continue;
      off_diag_max = std::max(off_diag_max, std::abs(H_interaction_rel(i, j)));
    }
  }

  CHECK(std::abs(diag_value - U_hubbard) < kRelTransformTolerance);
  CHECK(off_diag_max < kRelTransformTolerance);
}

TEST_CASE("relative_basis_transform_with_index") {
  const size_t lattice_size = 4;
  const std::vector<size_t> size{lattice_size};
  const size_t K = 1;

  Index index(size);
  Basis momentum_basis =
      Basis::with_fixed_particle_number_spin_momentum(lattice_size, 2, 0, index, {K});

  auto result = relative_position_transform_with_index(momentum_basis, index);

  CHECK(result.num_relative_coords == 1);
  CHECK(result.relative_index.size() == lattice_size);
  CHECK(result.transform.n_rows == lattice_size);
  CHECK(result.transform.n_cols == momentum_basis.set.size());
}

TEST_CASE("relative_basis_transform_different_momenta") {
  const size_t lattice_size = 6;
  const std::vector<size_t> size{lattice_size};
  const double t = 1.0;
  const double U_hubbard = 4.0;

  Index index(size);

  // Test for different total momenta K
  for (size_t K = 0; K < lattice_size; ++K) {
    Basis momentum_basis =
        Basis::with_fixed_particle_number_spin_momentum(lattice_size, 2, 0, index, {K});

    arma::cx_mat U = relative_position_transform(momentum_basis, index);

    // Check unitarity for each K
    const arma::cx_mat should_be_identity = U.t() * U;
    const arma::cx_mat identity = arma::eye<arma::cx_mat>(U.n_cols, U.n_cols);
    const double unitarity_error = arma::norm(should_be_identity - identity, "fro");
    CHECK(unitarity_error < kRelTransformTolerance);

    // Check eigenvalue preservation for each K
    HubbardModelMomentum hubbard(t, U_hubbard, size);
    arma::cx_mat H_mom =
        compute_matrix_elements<arma::cx_mat>(momentum_basis, hubbard.hamiltonian());
    const arma::cx_mat H_rel = U.t() * H_mom * U;

    arma::vec eig_mom;
    arma::eig_sym(eig_mom, arma::cx_mat(H_mom));

    arma::vec eig_rel;
    arma::eig_sym(eig_rel, arma::cx_mat(H_rel));

    const double eig_error = arma::norm(arma::sort(eig_mom) - arma::sort(eig_rel));
    CHECK(eig_error < kRelTransformTolerance);
  }
}
