#include <armadillo>
#include <catch2/catch.hpp>

#include "algebra/basis.h"
#include "algebra/matrix_elements.h"
#include "algebra/operator.h"
#include "algebra/term.h"

TEST_CASE("matrix_elements_vector_serial_creation") {
  Basis basis(1, 1, Basis::Strategy::Restrict);
  Expression A(Operator::creation(Operator::Spin::Up, 0));
  NormalOrderer orderer;

  arma::cx_vec vec = compute_vector_elements_serial<arma::cx_vec>(basis, A, orderer);

  CHECK((vec.n_elem) == (2u));
  CHECK((vec(0)) == (arma::cx_double(1.0, 0.0)));
  CHECK((vec(1)) == (arma::cx_double(0.0, 0.0)));
}

TEST_CASE("matrix_elements_vector_parallel_matches_serial") {
  Basis basis(1, 1, Basis::Strategy::Restrict);
  Expression A(Operator::creation(Operator::Spin::Up, 0));
  NormalOrderer orderer;

  arma::cx_vec serial = compute_vector_elements_serial<arma::cx_vec>(basis, A, orderer);
  arma::cx_vec parallel = compute_vector_elements<arma::cx_vec>(basis, A);

  CHECK((parallel.n_elem) == (serial.n_elem));
  CHECK((parallel(0)) == (serial(0)));
  CHECK((parallel(1)) == (serial(1)));
}

TEST_CASE("matrix_elements_matrix_serial_density") {
  Basis basis(1, 1, Basis::Strategy::Restrict);
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
  Basis basis(1, 1, Basis::Strategy::Restrict);
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
