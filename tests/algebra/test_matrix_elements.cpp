#include <armadillo>
#include <catch2/catch.hpp>
#include <vector>

#include "algebra/basis.h"
#include "algebra/matrix_elements.h"
#include "algebra/model/hubbard_model_momentum.h"
#include "algebra/operator.h"
#include "algebra/term.h"
#include "utils/index.h"

TEST_CASE("matrix_elements_vector_serial_creation") {
  Basis basis = Basis::with_fixed_particle_number(1, 1);
  Expression A(Operator::creation(Operator::Spin::Up, 0));
  NormalOrderer orderer;

  arma::cx_vec vec = compute_vector_elements_serial<arma::cx_vec>(basis, A, orderer);

  CHECK((vec.n_elem) == (2u));
  CHECK((vec(0)) == (arma::cx_double(1.0, 0.0)));
  CHECK((vec(1)) == (arma::cx_double(0.0, 0.0)));
}

TEST_CASE("matrix_elements_vector_parallel_matches_serial") {
  Basis basis = Basis::with_fixed_particle_number(1, 1);
  Expression A(Operator::creation(Operator::Spin::Up, 0));
  NormalOrderer orderer;

  arma::cx_vec serial = compute_vector_elements_serial<arma::cx_vec>(basis, A, orderer);
  arma::cx_vec parallel = compute_vector_elements<arma::cx_vec>(basis, A);

  CHECK((parallel.n_elem) == (serial.n_elem));
  CHECK((parallel(0)) == (serial(0)));
  CHECK((parallel(1)) == (serial(1)));
}

TEST_CASE("matrix_elements_matrix_serial_density") {
  Basis basis = Basis::with_fixed_particle_number(1, 1);
  Expression A(Term(
      {Operator::creation(Operator::Spin::Up, 0), Operator::annihilation(Operator::Spin::Up, 0)}));
  NormalOrderer orderer;

  arma::cx_mat mat = compute_matrix_elements_serial<arma::cx_mat>(basis, A, orderer);

  CHECK((mat.n_rows) == (2u));
  CHECK((mat.n_cols) == (2u));
  CHECK((mat(0, 0)) == (arma::cx_double(1.0, 0.0)));
  CHECK((mat(0, 1)) == (arma::cx_double(0.0, 0.0)));
  CHECK((mat(1, 0)) == (arma::cx_double(0.0, 0.0)));
  CHECK((mat(1, 1)) == (arma::cx_double(0.0, 0.0)));
}

TEST_CASE("matrix_elements_matrix_parallel_matches_serial") {
  Basis basis = Basis::with_fixed_particle_number(1, 1);
  Expression A(Term(
      {Operator::creation(Operator::Spin::Up, 0), Operator::annihilation(Operator::Spin::Up, 0)}));
  NormalOrderer orderer;

  arma::cx_mat serial = compute_matrix_elements_serial<arma::cx_mat>(basis, A, orderer);
  arma::cx_mat parallel = compute_matrix_elements<arma::cx_mat>(basis, A);

  CHECK((parallel.n_rows) == (serial.n_rows));
  CHECK((parallel.n_cols) == (serial.n_cols));
  CHECK((parallel(0, 0)) == (serial(0, 0)));
  CHECK((parallel(0, 1)) == (serial(0, 1)));
  CHECK((parallel(1, 0)) == (serial(1, 0)));
  CHECK((parallel(1, 1)) == (serial(1, 1)));
}

TEST_CASE("rectangular_matrix_elements_serial_current_operator") {
  const std::vector<size_t> size = {4};
  Index index(size);

  // Momentum sector K=0
  const std::vector<size_t> K = {0};
  Basis basis_K = Basis::with_fixed_particle_number_spin_momentum(4, 2, 0, index, K);

  // Momentum sector K+Q where Q=1
  const std::vector<size_t> KQ = {1};
  Basis basis_KQ = Basis::with_fixed_particle_number_spin_momentum(4, 2, 0, index, KQ);

  // Create current operator expression
  HubbardModelMomentum model(1.0, 4.0, size);
  const std::vector<size_t> Q = {1};
  Expression current_expr = model.current(Q, 0);

  NormalOrderer orderer;
  arma::cx_mat mat = compute_rectangular_matrix_elements_serial<arma::cx_mat>(
      basis_KQ, basis_K, current_expr, orderer);

  // Verify dimensions
  CHECK(mat.n_rows == basis_KQ.set.size());
  CHECK(mat.n_cols == basis_K.set.size());
}

TEST_CASE("rectangular_matrix_elements_parallel_matches_serial") {
  const std::vector<size_t> size = {4};
  Index index(size);

  // Momentum sector K=1
  const std::vector<size_t> K = {1};
  Basis basis_K = Basis::with_fixed_particle_number_spin_momentum(4, 2, 0, index, K);

  // Momentum sector K+Q where Q=2
  const std::vector<size_t> KQ = {3};
  Basis basis_KQ = Basis::with_fixed_particle_number_spin_momentum(4, 2, 0, index, KQ);

  // Create current operator expression
  HubbardModelMomentum model(1.0, 4.0, size);
  const std::vector<size_t> Q = {2};
  Expression current_expr = model.current(Q, 0);

  NormalOrderer orderer;
  arma::cx_mat serial = compute_rectangular_matrix_elements_serial<arma::cx_mat>(
      basis_KQ, basis_K, current_expr, orderer);
  arma::cx_mat parallel =
      compute_rectangular_matrix_elements<arma::cx_mat>(basis_KQ, basis_K, current_expr);

  CHECK(parallel.n_rows == serial.n_rows);
  CHECK(parallel.n_cols == serial.n_cols);

  for (size_t i = 0; i < serial.n_rows; ++i) {
    for (size_t j = 0; j < serial.n_cols; ++j) {
      CHECK(std::abs(parallel(i, j) - serial(i, j)) < 1e-10);
    }
  }
}

TEST_CASE("rectangular_matrix_elements_sparse_matrix") {
  const std::vector<size_t> size = {4};
  Index index(size);

  const std::vector<size_t> K = {0};
  Basis basis_K = Basis::with_fixed_particle_number_spin_momentum(4, 2, 0, index, K);

  const std::vector<size_t> KQ = {1};
  Basis basis_KQ = Basis::with_fixed_particle_number_spin_momentum(4, 2, 0, index, KQ);

  HubbardModelMomentum model(1.0, 4.0, size);
  const std::vector<size_t> Q = {1};
  Expression current_expr = model.current(Q, 0);

  // Test that sparse matrix version works
  arma::sp_cx_mat sparse_mat =
      compute_rectangular_matrix_elements<arma::sp_cx_mat>(basis_KQ, basis_K, current_expr);
  arma::cx_mat dense_mat =
      compute_rectangular_matrix_elements<arma::cx_mat>(basis_KQ, basis_K, current_expr);

  CHECK(sparse_mat.n_rows == dense_mat.n_rows);
  CHECK(sparse_mat.n_cols == dense_mat.n_cols);

  // Convert sparse to dense and compare
  arma::cx_mat sparse_as_dense(sparse_mat);
  for (size_t i = 0; i < dense_mat.n_rows; ++i) {
    for (size_t j = 0; j < dense_mat.n_cols; ++j) {
      CHECK(std::abs(sparse_as_dense(i, j) - dense_mat(i, j)) < 1e-10);
    }
  }
}
