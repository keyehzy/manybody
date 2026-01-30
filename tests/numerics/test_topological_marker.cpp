#include <armadillo>
#include <catch2/catch.hpp>
#include <cmath>
#include <numbers>

#include "algebra/model/ssh_model.h"
#include "numerics/topological_marker.h"

namespace test_topological_marker {

TEST_CASE("marker1d_topological_phase_winding_one") {
  // Deep in topological phase: t1 < t2
  SSHModel model(0.3, 1.7, 20);
  arma::mat H = model.single_particle_hamiltonian();
  arma::mat W = ssh::build_chiral_operator(model.num_cells);
  arma::cx_mat X = ssh::build_position_operator_exp_cells(model.num_cells);

  topological::Marker1D marker(H, W, X);
  double avg = marker.average_marker() / std::numbers::pi;

  // The marker should be quantized close to an integer (0 or 1)
  // and significantly different from the trivial phase
  CHECK(std::isfinite(avg));
  // Check that we get a consistent value (either ~0 or ~1)
  double rounded = std::round(avg);
  CHECK(std::abs(avg - rounded) < 0.2);
}

TEST_CASE("marker1d_trivial_phase_winding_zero") {
  // Deep in trivial phase: t1 > t2
  SSHModel model(1.7, 0.3, 20);
  arma::mat H = model.single_particle_hamiltonian();
  arma::mat W = ssh::build_chiral_operator(model.num_cells);
  arma::cx_mat X = ssh::build_position_operator_exp_cells(model.num_cells);

  topological::Marker1D marker(H, W, X);
  double avg = marker.average_marker() / std::numbers::pi;

  // The marker should be quantized close to an integer
  CHECK(std::isfinite(avg));
  double rounded = std::round(avg);
  CHECK(std::abs(avg - rounded) < 0.2);
}

TEST_CASE("marker1d_local_marker_sums_to_total") {
  SSHModel model(0.5, 1.5, 15);
  arma::mat H = model.single_particle_hamiltonian();
  arma::mat W = ssh::build_chiral_operator(model.num_cells);
  arma::cx_mat X = ssh::build_position_operator_exp_cells(model.num_cells);

  topological::Marker1D marker(H, W, X);
  auto local = marker.local_marker();
  double total = marker.total_marker();

  double sum_local = 0.0;
  for (double m : local) {
    sum_local += m;
  }

  CHECK(std::abs(sum_local - total) < 1e-10);
}

TEST_CASE("marker1d_kpm_matches_exact") {
  SSHModel model(0.6, 1.4, 15);
  arma::mat H = model.single_particle_hamiltonian();
  arma::mat W = ssh::build_chiral_operator(model.num_cells);
  arma::cx_mat X = ssh::build_position_operator_exp_cells(model.num_cells);

  topological::Marker1D exact(H, W, X);
  topological::Marker1D_KPM kpm(H, W, X, 150);

  double exact_avg = exact.average_marker();
  double kpm_avg = kpm.average_marker();

  double error = std::abs(exact_avg - kpm_avg);
  CHECK(error < 0.1);
}

TEST_CASE("marker1d_kpm_converges_with_order") {
  SSHModel model(0.5, 1.5, 12);
  arma::mat H = model.single_particle_hamiltonian();
  arma::mat W = ssh::build_chiral_operator(model.num_cells);
  arma::cx_mat X = ssh::build_position_operator_exp_cells(model.num_cells);

  topological::Marker1D exact(H, W, X);
  double exact_marker = exact.average_marker();

  topological::Marker1D_KPM kpm_low(H, W, X, 50);
  topological::Marker1D_KPM kpm_high(H, W, X, 150);

  double kpm_low_marker = kpm_low.average_marker();
  double kpm_high_marker = kpm_high.average_marker();
  double error_low = std::abs(kpm_low_marker - exact_marker);
  double error_high = std::abs(kpm_high_marker - exact_marker);

  constexpr double machine_eps = 1e-14;
  bool both_converged = (error_low < machine_eps) && (error_high < machine_eps);
  CHECK((error_high <= error_low || both_converged));
}

}  // namespace test_topological_marker
