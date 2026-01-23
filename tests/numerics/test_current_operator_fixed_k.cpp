#include <armadillo>
#include <catch2/catch.hpp>
#include <complex>
#include <cstddef>
#include <vector>

#include "algebra/basis.h"
#include "algebra/matrix_elements.h"
#include "algebra/model/hubbard_model_momentum.h"
#include "numerics/current_operator_fixed_k.h"
#include "utils/index.h"

namespace {
constexpr double kCurrentTolerance = 1e-10;

// Compute rectangular matrix elements for current operator (K -> K+Q)
arma::cx_mat compute_current_matrix(const Basis& source_basis, const Basis& target_basis,
                                    const Expression& current_expr) {
  const auto& row_set = target_basis.set;
  const auto& col_set = source_basis.set;
  arma::cx_mat result(row_set.size(), col_set.size(), arma::fill::zeros);

  NormalOrderer orderer;
  for (size_t j = 0; j < col_set.size(); ++j) {
    Expression right(col_set[j]);
    Expression product = orderer.normal_order(current_expr * right);
    for (const auto& term : product.hashmap) {
      if (row_set.contains(term.first)) {
        size_t i = row_set.index_of(term.first);
        result(i, j) = term.second;
      }
    }
  }
  return result;
}
}  // namespace

TEST_CASE("CurrentOperatorFixedK construction", "[current][fixed-k]") {
  const std::vector<size_t> size = {4};
  Index index(size);

  // Sector K=0
  const std::vector<size_t> K = {0};
  Basis basis_K = Basis::with_fixed_particle_number_spin_momentum(4, 2, 0, index, K);

  // Sector K+Q where Q=1
  const std::vector<size_t> KQ = {1};
  Basis basis_KQ = Basis::with_fixed_particle_number_spin_momentum(4, 2, 0, index, KQ);

  // Q as flat index
  size_t Q_flat = index(std::vector<size_t>{1});

  CurrentOperatorFixedK J(basis_K, basis_KQ, size, 1.0, Q_flat, 0);

  REQUIRE(J.source_dimension() == basis_K.set.size());
  REQUIRE(J.target_dimension() == basis_KQ.set.size());
}

TEST_CASE("CurrentOperatorFixedK matches dense matrix 1D", "[current][fixed-k]") {
  const std::vector<size_t> size = {4};
  Index index(size);
  const double t = 1.0;

  HubbardModelMomentum model(t, 4.0, size);

  // Test all momentum sectors and Q values
  for (size_t K_val = 0; K_val < 4; ++K_val) {
    for (size_t Q_val = 0; Q_val < 4; ++Q_val) {
      const std::vector<size_t> K = {K_val};
      const std::vector<size_t> Q = {Q_val};
      const std::vector<size_t> KQ = {(K_val + Q_val) % 4};

      Basis basis_K = Basis::with_fixed_particle_number_spin_momentum(4, 2, 0, index, K);
      Basis basis_KQ = Basis::with_fixed_particle_number_spin_momentum(4, 2, 0, index, KQ);

      if (basis_K.set.empty() || basis_KQ.set.empty()) {
        continue;
      }

      // Build dense matrix using Expression
      Expression current_expr = model.current(Q, 0);
      arma::cx_mat J_dense = compute_current_matrix(basis_K, basis_KQ, current_expr);

      // Build matrix-free operator
      size_t Q_flat = index(Q);
      CurrentOperatorFixedK J_op(basis_K, basis_KQ, size, t, Q_flat, 0);

      // Test apply on random vectors
      arma::cx_vec v(basis_K.set.size(), arma::fill::randn);
      arma::cx_vec w_dense = J_dense * v;
      arma::cx_vec w_mf = J_op.apply(v);

      REQUIRE(arma::approx_equal(w_dense, w_mf, "absdiff", kCurrentTolerance));
    }
  }
}

TEST_CASE("CurrentOperatorFixedK adjoint matches dense matrix 1D", "[current][fixed-k]") {
  const std::vector<size_t> size = {4};
  Index index(size);
  const double t = 1.0;

  HubbardModelMomentum model(t, 4.0, size);

  // Test a few momentum sectors
  for (size_t K_val = 0; K_val < 4; ++K_val) {
    for (size_t Q_val = 0; Q_val < 4; ++Q_val) {
      const std::vector<size_t> K = {K_val};
      const std::vector<size_t> Q = {Q_val};
      const std::vector<size_t> KQ = {(K_val + Q_val) % 4};

      Basis basis_K = Basis::with_fixed_particle_number_spin_momentum(4, 2, 0, index, K);
      Basis basis_KQ = Basis::with_fixed_particle_number_spin_momentum(4, 2, 0, index, KQ);

      if (basis_K.set.empty() || basis_KQ.set.empty()) {
        continue;
      }

      // Build dense matrix using Expression
      Expression current_expr = model.current(Q, 0);
      arma::cx_mat J_dense = compute_current_matrix(basis_K, basis_KQ, current_expr);
      arma::cx_mat J_adj_dense = J_dense.t();

      // Build matrix-free operator
      size_t Q_flat = index(Q);
      CurrentOperatorFixedK J_op(basis_K, basis_KQ, size, t, Q_flat, 0);

      // Test adjoint_apply on random vectors
      arma::cx_vec v(basis_KQ.set.size(), arma::fill::randn);
      arma::cx_vec w_dense = J_adj_dense * v;
      arma::cx_vec w_mf = J_op.adjoint_apply(v);

      REQUIRE(arma::approx_equal(w_dense, w_mf, "absdiff", kCurrentTolerance));
    }
  }
}

TEST_CASE("CurrentOperatorFixedK matches dense matrix 2D", "[current][fixed-k]") {
  const std::vector<size_t> size = {3, 3};
  Index index(size);
  const double t = 1.5;

  HubbardModelMomentum model(t, 4.0, size);

  // Test a few momentum sectors in both directions
  const std::vector<size_t> K = {1, 0};
  const std::vector<size_t> Q = {1, 1};
  const std::vector<size_t> KQ = {(K[0] + Q[0]) % 3, (K[1] + Q[1]) % 3};

  Basis basis_K = Basis::with_fixed_particle_number_spin_momentum(9, 2, 0, index, K);
  Basis basis_KQ = Basis::with_fixed_particle_number_spin_momentum(9, 2, 0, index, KQ);

  REQUIRE(!basis_K.set.empty());
  REQUIRE(!basis_KQ.set.empty());

  // Test both directions
  for (size_t direction = 0; direction < 2; ++direction) {
    Expression current_expr = model.current(Q, direction);
    arma::cx_mat J_dense = compute_current_matrix(basis_K, basis_KQ, current_expr);

    size_t Q_flat = index(Q);
    CurrentOperatorFixedK J_op(basis_K, basis_KQ, size, t, Q_flat, direction);

    // Test apply
    arma::cx_vec v(basis_K.set.size(), arma::fill::randn);
    arma::cx_vec w_dense = J_dense * v;
    arma::cx_vec w_mf = J_op.apply(v);

    REQUIRE(arma::approx_equal(w_dense, w_mf, "absdiff", kCurrentTolerance));

    // Test adjoint
    arma::cx_vec v2(basis_KQ.set.size(), arma::fill::randn);
    arma::cx_vec w_adj_dense = J_dense.t() * v2;
    arma::cx_vec w_adj_mf = J_op.adjoint_apply(v2);

    REQUIRE(arma::approx_equal(w_adj_dense, w_adj_mf, "absdiff", kCurrentTolerance));
  }
}

TEST_CASE("CurrentOperatorFixedK Q=0 is diagonal", "[current][fixed-k]") {
  const std::vector<size_t> size = {4};
  Index index(size);
  const double t = 1.0;

  const std::vector<size_t> K = {0};
  const std::vector<size_t> Q = {0};

  Basis basis_K = Basis::with_fixed_particle_number_spin_momentum(4, 2, 0, index, K);

  // For Q=0, source and target are the same sector
  size_t Q_flat = index(Q);
  CurrentOperatorFixedK J_op(basis_K, basis_K, size, t, Q_flat, 0);

  REQUIRE(J_op.source_dimension() == J_op.target_dimension());

  // Build dense matrix for comparison
  HubbardModelMomentum model(t, 4.0, size);
  Expression current_expr = model.current(Q, 0);
  arma::cx_mat J_dense = compute_current_matrix(basis_K, basis_K, current_expr);

  // Verify it's diagonal
  for (size_t i = 0; i < J_dense.n_rows; ++i) {
    for (size_t j = 0; j < J_dense.n_cols; ++j) {
      if (i != j) {
        REQUIRE(std::abs(J_dense(i, j)) < kCurrentTolerance);
      }
    }
  }

  // Test matrix-free matches
  arma::cx_vec v(basis_K.set.size(), arma::fill::randn);
  arma::cx_vec w_dense = J_dense * v;
  arma::cx_vec w_mf = J_op.apply(v);
  REQUIRE(arma::approx_equal(w_dense, w_mf, "absdiff", kCurrentTolerance));
}

TEST_CASE("CurrentOperatorFixedK adjoint consistency", "[current][fixed-k]") {
  // Test that <u, J v> = <J† u, v>
  const std::vector<size_t> size = {4};
  Index index(size);
  const double t = 1.0;

  const std::vector<size_t> K = {1};
  const std::vector<size_t> Q = {2};
  const std::vector<size_t> KQ = {3};

  Basis basis_K = Basis::with_fixed_particle_number_spin_momentum(4, 2, 0, index, K);
  Basis basis_KQ = Basis::with_fixed_particle_number_spin_momentum(4, 2, 0, index, KQ);

  REQUIRE(!basis_K.set.empty());
  REQUIRE(!basis_KQ.set.empty());

  size_t Q_flat = index(Q);
  CurrentOperatorFixedK J_op(basis_K, basis_KQ, size, t, Q_flat, 0);

  arma::cx_vec v(basis_K.set.size(), arma::fill::randn);
  arma::cx_vec u(basis_KQ.set.size(), arma::fill::randn);

  arma::cx_vec Jv = J_op.apply(v);
  arma::cx_vec Jadj_u = J_op.adjoint_apply(u);

  std::complex<double> inner1 = arma::cdot(u, Jv);
  std::complex<double> inner2 = arma::cdot(Jadj_u, v);

  REQUIRE(std::abs(inner1 - inner2) < kCurrentTolerance);
}

TEST_CASE("CurrentOperatorFixedK different particle numbers", "[current][fixed-k]") {
  const std::vector<size_t> size = {4};
  Index index(size);
  const double t = 1.0;

  // Test with 3 particles (spin up majority)
  const std::vector<size_t> K = {0};
  const std::vector<size_t> Q = {1};
  const std::vector<size_t> KQ = {1};

  Basis basis_K = Basis::with_fixed_particle_number_spin_momentum(4, 3, 1, index, K);
  Basis basis_KQ = Basis::with_fixed_particle_number_spin_momentum(4, 3, 1, index, KQ);

  if (basis_K.set.empty() || basis_KQ.set.empty()) {
    SUCCEED("No basis states for this configuration");
    return;
  }

  HubbardModelMomentum model(t, 4.0, size);
  Expression current_expr = model.current(Q, 0);
  arma::cx_mat J_dense = compute_current_matrix(basis_K, basis_KQ, current_expr);

  size_t Q_flat = index(Q);
  CurrentOperatorFixedK J_op(basis_K, basis_KQ, size, t, Q_flat, 0);

  arma::cx_vec v(basis_K.set.size(), arma::fill::randn);
  arma::cx_vec w_dense = J_dense * v;
  arma::cx_vec w_mf = J_op.apply(v);

  REQUIRE(arma::approx_equal(w_dense, w_mf, "absdiff", kCurrentTolerance));
}

TEST_CASE("CurrentOperatorFixedK velocity values", "[current][fixed-k]") {
  const std::vector<size_t> size = {4};
  Index index(size);
  const double t = 1.0;

  const std::vector<size_t> K = {0};
  const std::vector<size_t> KQ = {0};
  size_t Q_flat = 0;

  Basis basis_K = Basis::with_fixed_particle_number_spin_momentum(4, 2, 0, index, K);
  CurrentOperatorFixedK J_op(basis_K, basis_K, size, t, Q_flat, 0);

  const auto& velocities = J_op.velocities();
  REQUIRE(velocities.size() == 4);

  // v_d(k) = 2t * (2π/L) * sin(2πk/L)
  // For L=4, d=0:
  // k=0: sin(0) = 0
  // k=1: sin(π/2) = 1
  // k=2: sin(π) = 0
  // k=3: sin(3π/2) = -1
  const double factor = 2.0 * t * (2.0 * M_PI / 4.0);
  REQUIRE(std::abs(velocities[0] - 0.0) < kCurrentTolerance);
  REQUIRE(std::abs(velocities[1] - factor * 1.0) < kCurrentTolerance);
  REQUIRE(std::abs(velocities[2] - 0.0) < kCurrentTolerance);
  REQUIRE(std::abs(velocities[3] - factor * (-1.0)) < kCurrentTolerance);
}
