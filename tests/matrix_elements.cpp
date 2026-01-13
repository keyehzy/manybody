#include "algebra/matrix_elements.h"

#include <armadillo>

#include "algebra/basis.h"
#include "algebra/operator.h"
#include "algebra/term.h"
#include "framework.h"

TEST(matrix_elements_vector_serial_creation) {
  Basis basis(1, 1, Basis::Strategy::Restrict);
  Expression A(Operator::creation(Operator::Spin::Up, 0));
  NormalOrderer orderer;

  arma::cx_vec vec = compute_vector_elements_serial<arma::cx_vec>(basis, A, orderer);

  EXPECT_EQ(vec.n_elem, 2u);
  EXPECT_EQ(vec(0), arma::cx_double(1.0, 0.0));
  EXPECT_EQ(vec(1), arma::cx_double(0.0, 0.0));
}

TEST(matrix_elements_vector_parallel_matches_serial) {
  Basis basis(1, 1, Basis::Strategy::Restrict);
  Expression A(Operator::creation(Operator::Spin::Up, 0));
  NormalOrderer orderer;

  arma::cx_vec serial = compute_vector_elements_serial<arma::cx_vec>(basis, A, orderer);
  arma::cx_vec parallel = compute_vector_elements<arma::cx_vec>(basis, A);

  EXPECT_EQ(parallel.n_elem, serial.n_elem);
  EXPECT_EQ(parallel(0), serial(0));
  EXPECT_EQ(parallel(1), serial(1));
}

TEST(matrix_elements_matrix_serial_density) {
  Basis basis(1, 1, Basis::Strategy::Restrict);
  Expression A(Term(
      {Operator::creation(Operator::Spin::Up, 0), Operator::annihilation(Operator::Spin::Up, 0)}));
  NormalOrderer orderer;

  arma::cx_mat mat = compute_matrix_elements_serial<arma::cx_mat>(basis, A, orderer);

  EXPECT_EQ(mat.n_rows, 2u);
  EXPECT_EQ(mat.n_cols, 2u);
  EXPECT_EQ(mat(0, 0), arma::cx_double(1.0, 0.0));
  EXPECT_EQ(mat(0, 1), arma::cx_double(0.0, 0.0));
  EXPECT_EQ(mat(1, 0), arma::cx_double(0.0, 0.0));
  EXPECT_EQ(mat(1, 1), arma::cx_double(0.0, 0.0));
}

TEST(matrix_elements_matrix_parallel_matches_serial) {
  Basis basis(1, 1, Basis::Strategy::Restrict);
  Expression A(Term(
      {Operator::creation(Operator::Spin::Up, 0), Operator::annihilation(Operator::Spin::Up, 0)}));
  NormalOrderer orderer;

  arma::cx_mat serial = compute_matrix_elements_serial<arma::cx_mat>(basis, A, orderer);
  arma::cx_mat parallel = compute_matrix_elements<arma::cx_mat>(basis, A);

  EXPECT_EQ(parallel.n_rows, serial.n_rows);
  EXPECT_EQ(parallel.n_cols, serial.n_cols);
  EXPECT_EQ(parallel(0, 0), serial(0, 0));
  EXPECT_EQ(parallel(0, 1), serial(0, 1));
  EXPECT_EQ(parallel(1, 0), serial(1, 0));
  EXPECT_EQ(parallel(1, 1), serial(1, 1));
}
