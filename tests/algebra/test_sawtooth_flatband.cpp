#include <armadillo>
#include <catch2/catch.hpp>
#include <cmath>
#include <complex>
#include <vector>

#include "algebra/boson/basis.h"
#include "algebra/boson/model/sawtooth_model.h"
#include "algebra/matrix_elements.h"

namespace {

constexpr double kTolerance = 1e-10;
constexpr double kEigenvectorTolerance = 1e-9;

BosonExpression v_state_creation(const SawtoothHubbardModel& model, size_t cell) {
  const auto coeff = BosonExpression::complex_type(std::sqrt(2.0), 0.0);
  return BosonExpression(
             boson::creation(SawtoothHubbardModel::species, model.site_apex(cell, -1))) +
         coeff * BosonExpression(
                     boson::creation(SawtoothHubbardModel::species, model.site_base(cell))) +
         BosonExpression(boson::creation(SawtoothHubbardModel::species, model.site_apex(cell)));
}

std::vector<std::pair<size_t, size_t>> non_overlapping_v_state_pairs(size_t num_cells) {
  std::vector<std::pair<size_t, size_t>> pairs;
  for (size_t left = 0; left < num_cells; ++left) {
    for (size_t right = left + 1; right < num_cells; ++right) {
      const bool adjacent = right == left + 1;
      const bool wraps = left == 0 && right + 1 == num_cells;
      if (!adjacent && !wraps) {
        pairs.emplace_back(left, right);
      }
    }
  }
  return pairs;
}

size_t ground_state_degeneracy(const arma::vec& eigenvalues, double tolerance) {
  const double ground_energy = eigenvalues(0);
  size_t degeneracy = 0;
  while (degeneracy < eigenvalues.n_elem &&
         std::abs(eigenvalues(degeneracy) - ground_energy) < tolerance) {
    ++degeneracy;
  }
  return degeneracy;
}

double boson_basis_state_norm(const BosonBasis::key_type& state) {
  if (state.empty()) {
    return 1.0;
  }

  double norm = 1.0;
  size_t multiplicity = 1;
  for (size_t i = 1; i < state.size(); ++i) {
    if (state[i] == state[i - 1]) {
      ++multiplicity;
      continue;
    }
    norm *= std::tgamma(static_cast<double>(multiplicity + 1));
    multiplicity = 1;
  }
  norm *= std::tgamma(static_cast<double>(multiplicity + 1));
  return norm;
}

std::vector<double> boson_basis_norms(const BosonBasis& basis) {
  std::vector<double> norms;
  norms.reserve(basis.set.size());
  for (const auto& state : basis.set) {
    norms.push_back(boson_basis_state_norm(state));
  }
  return norms;
}

arma::cx_mat normalize_boson_matrix(const arma::cx_mat& matrix, const std::vector<double>& norms) {
  arma::cx_mat result = matrix;
  for (size_t row = 0; row < result.n_rows; ++row) {
    for (size_t col = 0; col < result.n_cols; ++col) {
      result(row, col) *= std::sqrt(norms[row] / norms[col]);
    }
  }
  return result;
}

arma::cx_vec normalize_boson_vector(const arma::cx_vec& vector, const std::vector<double>& norms) {
  arma::cx_vec result = vector;
  for (size_t i = 0; i < result.n_elem; ++i) {
    result(i) *= std::sqrt(norms[i]);
  }
  return result;
}

}  // namespace

TEST_CASE("sawtooth flatband exact diagonalization one particle", "[sawtooth][flatband][ed]") {
  constexpr size_t num_cells = 4;
  constexpr double t_base = 1.0;
  const double t_tooth = std::sqrt(2.0) * t_base;

  const SawtoothHubbardModel model(t_base, t_tooth, 0.0, num_cells);
  const BosonBasis basis = BosonBasis::with_fixed_particle_number_and_spin(model.num_sites, 1, 1);

  const arma::cx_mat hamiltonian =
      compute_matrix_elements<arma::cx_mat>(basis, model.hamiltonian());

  REQUIRE(hamiltonian.n_rows == 2 * num_cells);
  REQUIRE(hamiltonian.n_cols == 2 * num_cells);
  REQUIRE(arma::norm(hamiltonian - hamiltonian.t(), "fro") < kTolerance);

  arma::vec eigenvalues;
  REQUIRE(arma::eig_sym(eigenvalues, hamiltonian));

  const arma::vec expected = {-4.0, -2.0, -2.0, 0.0, 2.0, 2.0, 2.0, 2.0};
  REQUIRE(eigenvalues.n_elem == expected.n_elem);

  for (size_t i = 0; i < expected.n_elem; ++i) {
    CHECK(std::abs(eigenvalues(i) - expected(i)) < kTolerance);
  }
}

TEST_CASE("sawtooth flatband two bosons ground-state degeneracy matches non-overlapping V states",
          "[sawtooth][flatband][ed]") {
  constexpr size_t num_cells = 6;
  constexpr double t_base = -1.0;
  constexpr double u = 1.0;
  const double t_tooth = std::sqrt(2.0) * std::abs(t_base);

  const SawtoothHubbardModel model(t_base, t_tooth, u, num_cells);
  const BosonBasis one_particle_basis =
      BosonBasis::with_fixed_particle_number_and_spin(model.num_sites, 1, 1);
  const BosonBasis basis = BosonBasis::with_fixed_particle_number_and_spin(model.num_sites, 2, 2);
  const auto basis_norms = boson_basis_norms(basis);

  const arma::cx_mat one_particle_hamiltonian =
      compute_matrix_elements<arma::cx_mat>(one_particle_basis, model.kinetic());
  const arma::cx_mat hamiltonian_coordinates =
      compute_matrix_elements<arma::cx_mat>(basis, model.hamiltonian());
  const arma::cx_mat interaction_coordinates =
      compute_matrix_elements<arma::cx_mat>(basis, model.interaction());
  const arma::cx_mat hamiltonian = normalize_boson_matrix(hamiltonian_coordinates, basis_norms);
  const arma::cx_mat interaction = normalize_boson_matrix(interaction_coordinates, basis_norms);

  arma::vec one_particle_eigenvalues;
  arma::vec eigenvalues;
  REQUIRE(arma::eig_sym(one_particle_eigenvalues, one_particle_hamiltonian));
  REQUIRE(arma::norm(hamiltonian - hamiltonian.t(), "fro") < kEigenvectorTolerance);
  REQUIRE(arma::eig_sym(eigenvalues, hamiltonian));

  const double flat_band_energy = 2.0 * t_base;
  const double expected_ground_energy = 2.0 * flat_band_energy;
  CHECK(std::abs(eigenvalues(0) - expected_ground_energy) < kTolerance);

  const auto cls_pairs = non_overlapping_v_state_pairs(num_cells);
  CHECK(ground_state_degeneracy(eigenvalues, kTolerance) == cls_pairs.size());

  arma::cx_mat candidate_vectors(basis.set.size(), cls_pairs.size(), arma::fill::zeros);

  for (size_t column = 0; column < cls_pairs.size(); ++column) {
    const auto& [left, right] = cls_pairs[column];
    const BosonExpression state =
        canonicalize(v_state_creation(model, left) * v_state_creation(model, right));
    arma::cx_vec vector =
        normalize_boson_vector(compute_vector_elements<arma::cx_vec>(basis, state), basis_norms);
    const double norm = arma::norm(vector);

    REQUIRE(norm > kTolerance);
    vector /= norm;
    candidate_vectors.col(column) = vector;

    const arma::cx_vec residual = hamiltonian * vector - expected_ground_energy * vector;
    CHECK(arma::norm(residual) < kEigenvectorTolerance);

    const std::complex<double> interaction_energy = arma::cdot(vector, interaction * vector);
    CHECK(std::abs(interaction_energy) < kEigenvectorTolerance);
  }

  arma::vec singular_values;
  REQUIRE(arma::svd(singular_values, candidate_vectors));

  size_t independent_candidates = 0;
  for (double singular_value : singular_values) {
    if (singular_value > kTolerance) {
      ++independent_candidates;
    }
  }
  CHECK(independent_candidates == cls_pairs.size());
}
