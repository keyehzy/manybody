#include <catch2/catch.hpp>
#include <cmath>

#include "algorithms/brg/thermal.h"

TEST_CASE("use_zero_temperature returns true for T=0") {
  CHECK(brg::use_zero_temperature(0.0));
  CHECK(brg::use_zero_temperature(-1.0));
}

TEST_CASE("use_zero_temperature returns true for very small T") {
  CHECK(brg::use_zero_temperature(1e-16));
  CHECK(brg::use_zero_temperature(1e-20));
}

TEST_CASE("use_zero_temperature returns false for finite T") {
  CHECK_FALSE(brg::use_zero_temperature(0.1));
  CHECK_FALSE(brg::use_zero_temperature(1.0));
  CHECK_FALSE(brg::use_zero_temperature(1e-10));
}

TEST_CASE("compute_thermal_weights handles empty eigenvalues") {
  arma::vec empty;
  auto w = brg::compute_thermal_weights(empty, 1.0);
  CHECK(w.weights.n_elem == 0);
  CHECK(std::isinf(w.logZ));
}

TEST_CASE("compute_thermal_weights at zero temperature limit") {
  arma::vec evals = {0.0, 1.0, 2.0};
  double beta = 1e10;  // very large beta ~ zero T

  auto w = brg::compute_thermal_weights(evals, beta);

  CHECK(w.weights(0) == Approx(1.0).margin(1e-10));
  CHECK(w.weights(1) == Approx(0.0).margin(1e-10));
  CHECK(w.weights(2) == Approx(0.0).margin(1e-10));
  CHECK(w.free_energy == Approx(0.0).margin(1e-10));
}

TEST_CASE("compute_thermal_weights at high temperature limit") {
  arma::vec evals = {0.0, 1.0, 2.0};
  double beta = 1e-6;  // very small beta ~ high T

  auto w = brg::compute_thermal_weights(evals, beta);

  // At high T, all weights should be approximately equal
  double expected = 1.0 / 3.0;
  CHECK(w.weights(0) == Approx(expected).margin(1e-4));
  CHECK(w.weights(1) == Approx(expected).margin(1e-4));
  CHECK(w.weights(2) == Approx(expected).margin(1e-4));
}

TEST_CASE("compute_thermal_weights normalization") {
  arma::vec evals = {-1.5, 0.0, 0.5, 2.0};
  double beta = 2.0;

  auto w = brg::compute_thermal_weights(evals, beta);

  double sum = arma::accu(w.weights);
  CHECK(sum == Approx(1.0).margin(1e-12));
}

TEST_CASE("compute_thermal_weights free energy relation") {
  // F = -T log Z = E0 - T log(Z/e^{-beta E0})
  arma::vec evals = {-2.0, 0.0, 1.0};
  double T = 0.5;
  double beta = 1.0 / T;

  auto w = brg::compute_thermal_weights(evals, beta);

  // Verify logZ and free_energy consistency
  double F_from_logZ = -T * w.logZ;
  CHECK(w.free_energy == Approx(F_from_logZ).margin(1e-12));
}

TEST_CASE("compute_thermal_weights with degenerate ground state") {
  // Two-fold degenerate ground state
  arma::vec evals = {0.0, 0.0, 2.0};
  double beta = 10.0;  // moderately low T

  auto w = brg::compute_thermal_weights(evals, beta);

  // Ground state weights should be approximately equal
  CHECK(w.weights(0) == Approx(w.weights(1)).margin(1e-10));
  // Sum of ground state weights should be ~1
  CHECK(w.weights(0) + w.weights(1) == Approx(1.0).margin(1e-6));
}
