#include <catch2/catch.hpp>
#include <cereal/archives/binary.hpp>
#include <sstream>

#include "utils/arma_cereal.h"

TEST_CASE("arma_cereal_sp_mat_empty") {
  arma::sp_mat original(3, 4);

  std::stringstream ss;
  {
    cereal::BinaryOutputArchive oarchive(ss);
    oarchive(original);
  }

  arma::sp_mat loaded;
  {
    cereal::BinaryInputArchive iarchive(ss);
    iarchive(loaded);
  }

  CHECK(loaded.n_rows == original.n_rows);
  CHECK(loaded.n_cols == original.n_cols);
  CHECK(loaded.n_nonzero == original.n_nonzero);
}

TEST_CASE("arma_cereal_sp_mat_with_values") {
  arma::sp_mat original(5, 6);
  original(0, 0) = 1.5;
  original(2, 3) = -2.7;
  original(4, 5) = 3.14159;

  std::stringstream ss;
  {
    cereal::BinaryOutputArchive oarchive(ss);
    oarchive(original);
  }

  arma::sp_mat loaded;
  {
    cereal::BinaryInputArchive iarchive(ss);
    iarchive(loaded);
  }

  CHECK(loaded.n_rows == original.n_rows);
  CHECK(loaded.n_cols == original.n_cols);
  CHECK(loaded.n_nonzero == original.n_nonzero);
  CHECK(arma::approx_equal(arma::mat(loaded), arma::mat(original), "absdiff", 1e-15));
}

TEST_CASE("arma_cereal_sp_cx_mat_with_values") {
  arma::sp_cx_mat original(4, 4);
  original(0, 1) = std::complex<double>(1.0, 2.0);
  original(2, 3) = std::complex<double>(-3.5, 4.5);
  original(3, 0) = std::complex<double>(0.0, -1.0);

  std::stringstream ss;
  {
    cereal::BinaryOutputArchive oarchive(ss);
    oarchive(original);
  }

  arma::sp_cx_mat loaded;
  {
    cereal::BinaryInputArchive iarchive(ss);
    iarchive(loaded);
  }

  CHECK(loaded.n_rows == original.n_rows);
  CHECK(loaded.n_cols == original.n_cols);
  CHECK(loaded.n_nonzero == original.n_nonzero);
  CHECK(arma::approx_equal(arma::cx_mat(loaded), arma::cx_mat(original), "absdiff", 1e-15));
}

TEST_CASE("arma_cereal_sp_mat_large_sparse") {
  arma::sp_mat original(100, 100);
  original(0, 99) = 1.0;
  original(50, 50) = 2.0;
  original(99, 0) = 3.0;

  std::stringstream ss;
  {
    cereal::BinaryOutputArchive oarchive(ss);
    oarchive(original);
  }

  arma::sp_mat loaded;
  {
    cereal::BinaryInputArchive iarchive(ss);
    iarchive(loaded);
  }

  CHECK(loaded.n_rows == 100);
  CHECK(loaded.n_cols == 100);
  CHECK(loaded.n_nonzero == 3);
  CHECK(loaded(0, 99) == Approx(1.0));
  CHECK(loaded(50, 50) == Approx(2.0));
  CHECK(loaded(99, 0) == Approx(3.0));
}
