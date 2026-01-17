#include <armadillo>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <vector>

#include "numerics/hubbard_relative_operators.h"
#include "numerics/linear_operator.h"

arma::cx_mat build_operator_matrix(const LinearOperator<arma::cx_vec>& op, size_t dimension) {
  arma::cx_mat m(dimension, dimension, arma::fill::zeros);
  arma::cx_vec basis_vector(dimension, arma::fill::zeros);
  for (size_t j = 0; j < dimension; ++j) {
    basis_vector.zeros();
    basis_vector(j) = 1.0;
    m.col(j) = op.apply(basis_vector);
  }
  return m;
}

double test_adjoint_property(const std::vector<size_t>& lattice_size, double t, size_t direction,
                             const std::vector<int64_t>& K, const std::vector<int64_t>& q) {
  const CurrentRelative_Q j_plus(lattice_size, t, K, q, direction);
  const CurrentRelative_Q_Adjoint j_minus(lattice_size, t, K, q, direction);

  const size_t dim = j_plus.dimension();
  const arma::cx_mat j_plus_mat = build_operator_matrix(j_plus, dim);
  const arma::cx_mat j_minus_mat = build_operator_matrix(j_minus, dim);
  const arma::cx_mat j_plus_adj = j_plus_mat.t();

  const double diff_norm = arma::norm(j_minus_mat - j_plus_adj, "fro");
  const double j_plus_norm = arma::norm(j_plus_mat, "fro");

  return j_plus_norm > 0 ? diff_norm / j_plus_norm : diff_norm;
}

int main() {
  const std::vector<size_t> lattice_1d{4};
  const double t = 1.0;
  const size_t direction = 0;

  std::cout << "=== Numerical Tests: CurrentRelative_Q_Adjoint vs J(q)† ===\n\n";

  // 1D test cases
  std::vector<std::pair<std::vector<int64_t>, std::vector<int64_t>>> cases_1d = {
      {{0}, {0}}, {{0}, {1}}, {{0}, {2}}, {{1}, {1}}, {{1}, {-1}}, {{2}, {2}},
  };

  std::cout << "1D Tests (L=4):\n";
  std::cout << "  K    q    RelError\n";
  std::cout << "  --------------------------------\n";
  for (const auto& [K, q] : cases_1d) {
    const double err = test_adjoint_property(lattice_1d, t, direction, K, q);
    printf("  %2lld   %2lld   %9.5e\n", K[0], q[0], err);
  }

  std::cout << "\n=== Detailed matrices for K=0, q=1 (1D, L=4) ===\n\n";

  const std::vector<int64_t> K0 = {0};
  const std::vector<int64_t> q1 = {1};

  CurrentRelative_Q j_plus(lattice_1d, t, K0, q1, 0);
  CurrentRelative_Q_Adjoint j_minus(lattice_1d, t, K0, q1, 0);

  arma::cx_mat J_plus = build_operator_matrix(j_plus, 4);
  arma::cx_mat J_minus = build_operator_matrix(j_minus, 4);
  arma::cx_mat J_plus_adj = J_plus.t();

  std::cout << "J(q) [K=0, q=1] (real part):\n" << arma::real(J_plus) << "\n";
  std::cout << "J(q) [K=0, q=1] (imag part):\n" << arma::imag(J_plus) << "\n";
  std::cout << "J(-q) - J(q)† (should be zero):\n" << (J_minus - J_plus_adj) << "\n";

  return 0;
}
