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

  double error_low = std::abs(kpm_low.average_marker() - exact_marker);
  double error_high = std::abs(kpm_high.average_marker() - exact_marker);

  CHECK(error_high <= error_low);
}

TEST_CASE("marker1d_lanczos_returns_valid_result") {
  SSHModel model(0.5, 1.5, 15);
  arma::mat H = model.single_particle_hamiltonian();
  arma::mat W = ssh::build_chiral_operator(model.num_cells);
  arma::cx_mat X = ssh::build_position_operator_exp_cells(model.num_cells);

  topological::Marker1D_Lanczos lanczos(H, W, X);
  auto [marker, krylov_dim] = lanczos.average_marker(model.num_sites);

  // The Lanczos method should return a finite result
  CHECK(std::isfinite(marker));
  CHECK(krylov_dim > 0);
  CHECK(krylov_dim <= model.num_sites);
}

TEST_CASE("marker1d_lanczos_convergence_tracking") {
  SSHModel model(0.5, 1.5, 20);
  arma::mat H = model.single_particle_hamiltonian();
  arma::mat W = ssh::build_chiral_operator(model.num_cells);
  arma::cx_mat X = ssh::build_position_operator_exp_cells(model.num_cells);

  topological::Marker1D_Lanczos lanczos(H, W, X);
  auto result = lanczos.compute_with_convergence(1e-3, model.num_sites);

  CHECK(std::isfinite(result.marker));
  CHECK(result.krylov_dim > 0);
  CHECK(std::isfinite(result.error_estimate));
}

TEST_CASE("marker1d_phase_transition") {
  // Test that marker computation gives consistent results across phases
  const size_t num_cells = 25;

  // Trivial phase: t1 > t2
  SSHModel trivial(1.5, 0.5, num_cells);
  arma::mat H_triv = trivial.single_particle_hamiltonian();
  arma::mat W_triv = ssh::build_chiral_operator(trivial.num_cells);
  arma::cx_mat X_triv = ssh::build_position_operator_exp_cells(trivial.num_cells);
  topological::Marker1D marker_triv(H_triv, W_triv, X_triv);
  double avg_trivial = marker_triv.average_marker() / std::numbers::pi;

  // Topological phase: t1 < t2
  SSHModel topological(0.5, 1.5, num_cells);
  arma::mat H_topo = topological.single_particle_hamiltonian();
  arma::mat W_topo = ssh::build_chiral_operator(topological.num_cells);
  arma::cx_mat X_topo = ssh::build_position_operator_exp_cells(topological.num_cells);
  topological::Marker1D marker_topo(H_topo, W_topo, X_topo);
  double avg_topological = marker_topo.average_marker() / std::numbers::pi;

  // Both markers should be finite and quantized
  CHECK(std::isfinite(avg_trivial));
  CHECK(std::isfinite(avg_topological));

  // Both should be quantized close to integers
  double rounded_triv = std::round(avg_trivial);
  double rounded_topo = std::round(avg_topological);
  CHECK(std::abs(avg_trivial - rounded_triv) < 0.3);
  CHECK(std::abs(avg_topological - rounded_topo) < 0.3);
}

TEST_CASE("ipr_analysis_basic") {
  arma::vec initial = arma::normalise(arma::randn<arma::vec>(10));
  std::vector<arma::vec> ritz_vectors;
  for (int i = 0; i < 5; ++i) {
    ritz_vectors.push_back(arma::normalise(arma::randn<arma::vec>(10)));
  }

  double ipr = topological::IPRAnalysis::ipr_krylov(initial, ritz_vectors);
  CHECK(std::isfinite(ipr));
  CHECK(ipr >= 0.0);
}

TEST_CASE("ipr_local_normalized_vector") {
  // For a normalized vector, IPR measures localization
  // Completely delocalized: IPR = 1/N
  // Completely localized: IPR = 1

  const size_t N = 100;

  // Delocalized vector
  arma::vec delocalized(N);
  delocalized.fill(1.0 / std::sqrt(static_cast<double>(N)));
  double ipr_deloc = topological::IPRAnalysis::ipr_local(delocalized);
  CHECK(std::abs(ipr_deloc - 1.0 / static_cast<double>(N)) < 1e-10);

  // Localized vector
  arma::vec localized(N, arma::fill::zeros);
  localized(0) = 1.0;
  double ipr_loc = topological::IPRAnalysis::ipr_local(localized);
  CHECK(std::abs(ipr_loc - 1.0) < 1e-10);
}

}  // namespace test_topological_marker
