#include <armadillo>
#include <catch2/catch.hpp>
#include <complex>

#include "algebra/boson/basis.h"
#include "algebra/boson/model/hubbard_model.h"
#include "algebra/matrix_elements.h"
#include "algorithms/static_susceptibility.h"

namespace {

constexpr double kTolerance = 1e-10;

bool complex_near(const std::complex<double>& lhs, const std::complex<double>& rhs) {
  return std::abs(lhs - rhs) < kTolerance;
}

}  // namespace

TEST_CASE("bose_hubbard_2d_basic_properties") {
  const BoseHubbardModel2D model(1.25, 3.5, 3, 4);

  CHECK(model.t == Approx(1.25));
  CHECK(model.u == Approx(3.5));
  CHECK(model.size_x == 3u);
  CHECK(model.size_y == 4u);
  CHECK(model.num_sites == 12u);
  CHECK(model.site(0, 0) == 0u);
  CHECK(model.site(1, 0) == 1u);
  CHECK(model.site(0, 1) == 3u);
}

TEST_CASE("bose_hubbard_2d_term_counts_match_square_lattice") {
  const BoseHubbardModel2D model(1.0, 4.0, 3, 3);

  CHECK(model.kinetic().size() == 36u);
  CHECK(model.interaction().size() == 18u);
  CHECK(model.hamiltonian().size() == 54u);
  CHECK(model.current(0).size() == 18u);
  CHECK(model.current(1).size() == 18u);
}

TEST_CASE("bose_hubbard_2d_current_contains_expected_oriented_bond_terms") {
  const double t = 1.5;
  const BoseHubbardModel2D model(t, 4.0, 3, 3);
  const BosonExpression current = model.current(0);

  const size_t left = model.site(0, 0);
  const size_t right = model.site(1, 0);

  BosonExpression::container_type forward{
      BosonOperator::creation(BoseHubbardModel2D::species, left),
      BosonOperator::annihilation(BoseHubbardModel2D::species, right),
  };
  BosonExpression::container_type backward{
      BosonOperator::creation(BoseHubbardModel2D::species, right),
      BosonOperator::annihilation(BoseHubbardModel2D::species, left),
  };

  const auto forward_it = current.terms().find(forward);
  const auto backward_it = current.terms().find(backward);

  REQUIRE(forward_it != current.terms().end());
  REQUIRE(backward_it != current.terms().end());
  CHECK(complex_near(forward_it->second, {0.0, -t}));
  CHECK(complex_near(backward_it->second, {0.0, t}));
}

TEST_CASE("bose_hubbard_2d_hamiltonian_and_current_are_hermitian") {
  const BoseHubbardModel2D model(1.0, 2.0, 2, 2);
  const BosonBasis basis = BosonBasis::with_fixed_particle_number_and_spin(model.num_sites, 2, 2);

  const arma::cx_mat hamiltonian =
      compute_matrix_elements<arma::cx_mat>(basis, model.hamiltonian());
  const arma::cx_mat current = compute_matrix_elements<arma::cx_mat>(basis, model.current(0));

  CHECK(arma::norm(hamiltonian - hamiltonian.t(), "fro") < kTolerance);
  CHECK(arma::norm(current - current.t(), "fro") < kTolerance);
}

TEST_CASE("zero_temperature_static_susceptibility_uses_spectral_integral") {
  const arma::vec eigenvalues = {0.0, 2.0};
  arma::cx_mat eigenvectors(2, 2, arma::fill::eye);
  arma::cx_mat observable(2, 2, arma::fill::zeros);
  observable(0, 1) = 3.0;
  observable(1, 0) = 3.0;

  const StaticSusceptibilityResult result =
      compute_zero_temperature_static_susceptibility(eigenvalues, eigenvectors, observable);

  CHECK(result.ground_energy == Approx(0.0));
  CHECK(result.value == Approx(4.5));
  CHECK(result.skipped_states == 0u);
}
