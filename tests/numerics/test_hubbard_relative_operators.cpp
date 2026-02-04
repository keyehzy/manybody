#include <armadillo>
#include <catch2/catch.hpp>
#include <cmath>
#include <numbers>
#include <vector>

#include "numerics/hubbard_relative_operators.h"

namespace {
constexpr double kTolerance = 1e-10;
}

TEST_CASE("hubbard_relative_interaction_keeps_origin") {
  HubbardRelativeInteraction interaction({4});
  arma::cx_vec v(4, arma::fill::zeros);
  v(0) = arma::cx_double(1.0, -0.5);
  v(1) = arma::cx_double(2.0, 0.0);
  v(2) = arma::cx_double(-3.0, 1.0);
  v(3) = arma::cx_double(0.25, -1.5);

  arma::cx_vec w = interaction.apply(v);

  CHECK(w(0) == v(0));
  CHECK(w(1) == arma::cx_double(0.0, 0.0));
  CHECK(w(2) == arma::cx_double(0.0, 0.0));
  CHECK(w(3) == arma::cx_double(0.0, 0.0));
}

TEST_CASE("hubbard_relative_kinetic_applies_periodic_neighbors") {
  const std::vector<size_t> size{4};
  const std::vector<int64_t> momentum{0};
  HubbardRelativeKinetic kinetic(size, momentum);

  arma::cx_vec v(4, arma::fill::zeros);
  v(0) = arma::cx_double(1.0, 0.0);
  arma::cx_vec w = kinetic.apply(v);

  const arma::cx_double expected(-2.0, 0.0);
  CHECK(std::abs(w(1) - expected) < kTolerance);
  CHECK(std::abs(w(3) - expected) < kTolerance);
  CHECK(std::abs(w(0)) < kTolerance);
  CHECK(std::abs(w(2)) < kTolerance);
}

TEST_CASE("hubbard_relative_combines_kinetic_and_interaction") {
  const std::vector<size_t> size{4};
  const std::vector<int64_t> momentum{0};
  HubbardRelative hubbard(size, momentum, 0.5, 3.0);

  arma::cx_vec v(4, arma::fill::zeros);
  v(0) = arma::cx_double(1.0, 0.0);
  arma::cx_vec w = hubbard.apply(v);

  CHECK(std::abs(w(0) - arma::cx_double(3.0, 0.0)) < kTolerance);
  CHECK(std::abs(w(1) - arma::cx_double(-1.0, 0.0)) < kTolerance);
  CHECK(std::abs(w(3) - arma::cx_double(-1.0, 0.0)) < kTolerance);
  CHECK(std::abs(w(2)) < kTolerance);
}

TEST_CASE("hubbard_relative_current_matches_neighbor_sum") {
  const std::vector<size_t> size{4};
  const std::vector<int64_t> momentum{1};
  constexpr double t = 2.0;
  HubbardRelativeCurrent current(size, momentum, t, 0);

  arma::cx_vec v(4, arma::fill::zeros);
  v(0) = arma::cx_double(1.0, 0.0);
  arma::cx_vec w = current.apply(v);

  const double k_phase = 2.0 * std::numbers::pi_v<double> * 1.0 / 4.0;
  const arma::cx_double expected(2.0 * t * std::sin(0.5 * k_phase), 0.0);
  CHECK(std::abs(w(1) - expected) < kTolerance);
  CHECK(std::abs(w(3) - expected) < kTolerance);
  CHECK(std::abs(w(0)) < kTolerance);
  CHECK(std::abs(w(2)) < kTolerance);
}

TEST_CASE("current_relative_q_adjoint_matches_inner_product") {
  const std::vector<size_t> size{4};
  const std::vector<int64_t> total_momentum{1};
  const std::vector<int64_t> transfer_momentum{2};
  constexpr double t = 1.1;

  CurrentRelative_Q current(size, t, total_momentum, transfer_momentum, 0);
  CurrentRelative_Q_Adjoint current_adj(size, t, total_momentum, transfer_momentum, 0);

  arma::cx_vec x(4);
  arma::cx_vec y(4);
  for (size_t i = 0; i < 4; ++i) {
    x(i) = arma::cx_double(0.5 + static_cast<double>(i), -0.1 * static_cast<double>(i));
    y(i) = arma::cx_double(1.2 - 0.3 * static_cast<double>(i), 0.2 + 0.1 * static_cast<double>(i));
  }

  const arma::cx_double left = arma::cdot(x, current.apply(y));
  const arma::cx_double right = arma::cdot(current_adj.apply(x), y);

  CHECK(std::abs(left - right) < kTolerance);
}
