#include <armadillo>
#include <catch2/catch.hpp>
#include <cmath>
#include <iostream>
#include <vector>

#include "algebra/boson/basis.h"
#include "algebra/boson/model/sawtooth_model.h"
#include "algebra/matrix_elements.h"

namespace {

constexpr double kTolerance = 1e-10;

}  // namespace

TEST_CASE("temporary sawtooth flatband exact diagonalization one particle",
          "[sawtooth][flatband][ed]") {
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
  std::cout << "Sawtooth flat-band eigenvalues: " << eigenvalues.t();

  const arma::vec expected = {-4.0, -2.0, -2.0, 0.0, 2.0, 2.0, 2.0, 2.0};
  REQUIRE(eigenvalues.n_elem == expected.n_elem);

  for (size_t i = 0; i < expected.n_elem; ++i) {
    CHECK(std::abs(eigenvalues(i) - expected(i)) < kTolerance);
  }
}
