#include <armadillo>
#include <catch2/catch.hpp>
#include <cmath>
#include <numbers>

#include "algebra/model/ssh_model.h"

namespace test_ssh_model {

TEST_CASE("ssh_model_basic_properties") {
  SSHModel model(0.5, 1.5, 10);

  CHECK(model.t1 == 0.5);
  CHECK(model.t2 == 1.5);
  CHECK(model.num_cells == 10);
  CHECK(model.num_sites == 20);
}

TEST_CASE("ssh_model_bulk_gap") {
  // Gap = 2 * |t1 - t2|
  SSHModel model1(0.5, 1.5, 10);
  CHECK(std::abs(model1.bulk_gap() - 2.0) < 1e-10);

  SSHModel model2(1.0, 1.0, 10);
  CHECK(std::abs(model2.bulk_gap()) < 1e-10);  // Critical point

  SSHModel model3(1.5, 0.5, 10);
  CHECK(std::abs(model3.bulk_gap() - 2.0) < 1e-10);
}

TEST_CASE("ssh_model_winding_number") {
  // t1 < t2: topological, winding = 1
  SSHModel topological(0.5, 1.5, 10);
  CHECK(topological.winding_number() == 1);

  // t1 > t2: trivial, winding = 0
  SSHModel trivial(1.5, 0.5, 10);
  CHECK(trivial.winding_number() == 0);

  // t1 = t2: critical, undefined
  SSHModel critical(1.0, 1.0, 10);
  CHECK(critical.winding_number() == -1);
}

TEST_CASE("ssh_model_hamiltonian_hermitian") {
  SSHModel model(0.7, 1.3, 8);
  arma::mat H = model.single_particle_hamiltonian();

  // Check Hermitian symmetry
  double asym = arma::norm(H - H.t(), "fro");
  CHECK(asym < 1e-12);
}

TEST_CASE("ssh_model_hamiltonian_obc_hermitian") {
  SSHModel model(0.7, 1.3, 8);
  arma::mat H = model.single_particle_hamiltonian_obc();

  double asym = arma::norm(H - H.t(), "fro");
  CHECK(asym < 1e-12);
}

TEST_CASE("ssh_model_pbc_spectrum_particle_hole_symmetric") {
  SSHModel model(0.6, 1.4, 12);
  arma::mat H = model.single_particle_hamiltonian();
  arma::vec eigvals = arma::eig_sym(H);

  // SSH model has particle-hole symmetry: for each E, there is -E
  arma::vec sorted = arma::sort(eigvals);
  for (size_t i = 0; i < sorted.n_elem / 2; ++i) {
    double E_neg = sorted(i);
    double E_pos = sorted(sorted.n_elem - 1 - i);
    CHECK(std::abs(E_neg + E_pos) < 1e-10);
  }
}

TEST_CASE("ssh_model_dispersion_matches_spectrum") {
  const size_t num_cells = 20;
  SSHModel model(0.6, 1.4, num_cells);
  arma::mat H = model.single_particle_hamiltonian();
  arma::vec eigvals = arma::sort(arma::eig_sym(H));

  // For PBC, allowed k values are 2*pi*n/L
  std::vector<double> k_values;
  for (size_t n = 0; n < num_cells; ++n) {
    k_values.push_back(2.0 * std::numbers::pi * static_cast<double>(n) /
                       static_cast<double>(num_cells));
  }

  std::vector<double> expected_E;
  for (double k : k_values) {
    auto [E_lower, E_upper] = model.dispersion(k);
    expected_E.push_back(E_lower);
    expected_E.push_back(E_upper);
  }
  std::sort(expected_E.begin(), expected_E.end());

  for (size_t i = 0; i < eigvals.n_elem; ++i) {
    CHECK(std::abs(eigvals(i) - expected_E[i]) < 1e-10);
  }
}

TEST_CASE("ssh_model_obc_has_edge_states_in_topological_phase") {
  // In topological phase with OBC, there should be zero-energy edge states
  SSHModel model(0.3, 1.7, 20);  // Deep in topological phase
  arma::mat H = model.single_particle_hamiltonian_obc();
  arma::vec eigvals = arma::sort(arma::eig_sym(H));

  // There should be two eigenvalues very close to zero (edge states)
  double gap = model.bulk_gap();
  size_t zero_energy_count = 0;
  for (size_t i = 0; i < eigvals.n_elem; ++i) {
    if (std::abs(eigvals(i)) < gap / 4.0) {
      zero_energy_count++;
    }
  }
  CHECK(zero_energy_count == 2);
}

TEST_CASE("ssh_model_obc_no_edge_states_in_trivial_phase") {
  // In trivial phase with OBC, no edge states
  SSHModel model(1.7, 0.3, 20);  // Deep in trivial phase
  arma::mat H = model.single_particle_hamiltonian_obc();
  arma::vec eigvals = arma::sort(arma::eig_sym(H));

  // No eigenvalues should be near zero
  double gap = model.bulk_gap();
  for (size_t i = 0; i < eigvals.n_elem; ++i) {
    CHECK(std::abs(eigvals(i)) > gap / 4.0);
  }
}

TEST_CASE("ssh_chiral_operator_properties") {
  const size_t num_cells = 5;
  arma::mat W = ssh::build_chiral_operator(num_cells);

  // W should be diagonal with +1 on A sublattice, -1 on B sublattice
  for (size_t n = 0; n < num_cells; ++n) {
    CHECK(W(2 * n, 2 * n) == 1.0);
    CHECK(W(2 * n + 1, 2 * n + 1) == -1.0);
  }

  // W^2 = I
  arma::mat W_squared = W * W;
  double error = arma::norm(W_squared - arma::eye(2 * num_cells, 2 * num_cells), "fro");
  CHECK(error < 1e-12);

  // Tr(W) = 0 (equal number of +1 and -1)
  CHECK(std::abs(arma::trace(W)) < 1e-12);
}

TEST_CASE("ssh_chiral_symmetry_holds") {
  SSHModel model(0.6, 1.4, 8);
  arma::mat H = model.single_particle_hamiltonian();
  arma::mat W = ssh::build_chiral_operator(model.num_cells);

  // Chiral symmetry: W H W = -H
  arma::mat WHW = W * H * W;
  double error = arma::norm(WHW + H, "fro");
  CHECK(error < 1e-12);
}

TEST_CASE("ssh_position_operator_cell_coordinates") {
  const size_t num_cells = 5;
  arma::mat X = ssh::build_position_operator_cells(num_cells);

  // A and B sites in the same unit cell should have the same position
  for (size_t n = 0; n < num_cells; ++n) {
    CHECK(X(2 * n, 2 * n) == static_cast<double>(n));
    CHECK(X(2 * n + 1, 2 * n + 1) == static_cast<double>(n));
  }
}

TEST_CASE("ssh_projector_correct_occupation") {
  SSHModel model(0.6, 1.4, 10);
  arma::mat H = model.single_particle_hamiltonian();
  arma::mat P = ssh::build_projector(H, 0.0);

  // At half-filling (E_F = 0), half the states are occupied
  double trace = arma::trace(P);
  CHECK(std::abs(trace - model.num_cells) < 1e-10);

  // P is idempotent: P^2 = P
  arma::mat P_squared = P * P;
  double error = arma::norm(P_squared - P, "fro");
  CHECK(error < 1e-10);
}

TEST_CASE("ssh_correlation_length_diverges_at_critical_point") {
  SSHModel far_from_critical(0.5, 1.5, 10);
  SSHModel near_critical(0.9, 1.1, 10);
  SSHModel critical(1.0, 1.0, 10);

  CHECK(near_critical.correlation_length() > far_from_critical.correlation_length());
  CHECK(std::isinf(critical.correlation_length()));
}

}  // namespace test_ssh_model
