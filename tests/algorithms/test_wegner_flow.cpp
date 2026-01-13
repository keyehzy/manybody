#include <armadillo>
#include <cmath>
#include <cstddef>

#include "algorithms/dynamical_system.h"
#include "algorithms/wegner_flow.h"
#include "framework.h"

namespace {

struct LinearDecay {
  double operator()(double /*t*/, double value) const { return -value; }
};

double off_diagonal_norm(const arma::cx_mat& H) {
  arma::cx_mat off = H;
  off.diag().zeros();
  return arma::norm(off, "fro");
}

double off_block_norm(const arma::cx_mat& H, size_t p_dim) {
  arma::cx_mat off(H.n_rows, H.n_cols, arma::fill::zeros);
  if (p_dim == 0 || p_dim >= H.n_rows) {
    return arma::norm(off, "fro");
  }

  off.submat(0, p_dim, p_dim - 1, H.n_cols - 1) = H.submat(0, p_dim, p_dim - 1, H.n_cols - 1);
  off.submat(p_dim, 0, H.n_rows - 1, p_dim - 1) = H.submat(p_dim, 0, H.n_rows - 1, p_dim - 1);
  return arma::norm(off, "fro");
}

}  // namespace

TEST(integrator_euler_matches_decay_solution) {
  const double result = integrate(LinearDecay{}, 1.0, 0.0, 1.0, 0.001, IntegratorMethod::kEuler);
  EXPECT_TRUE(std::abs(result - std::exp(-1.0)) < 5e-3);
}

TEST(integrator_rk4_matches_decay_solution) {
  const double result =
      integrate(LinearDecay{}, 1.0, 0.0, 1.0, 0.01, IntegratorMethod::kRungeKutta4);
  EXPECT_TRUE(std::abs(result - std::exp(-1.0)) < 1e-6);
}

TEST(wegner_flow_diagonalizes_hermitian_matrix) {
  arma::cx_mat H0(2, 2, arma::fill::zeros);
  H0(0, 0) = 1.0;
  H0(1, 1) = 2.0;
  H0(0, 1) = 0.5;
  H0(1, 0) = 0.5;

  const double off0 = off_diagonal_norm(H0);
  const arma::cx_mat H_final = wegner_flow(H0, 5.0, 0.01, IntegratorMethod::kRungeKutta4);
  const double off1 = off_diagonal_norm(H_final);

  EXPECT_TRUE(off1 < off0);
  EXPECT_TRUE(off1 < 1e-4);

  arma::vec vals0;
  arma::vec vals1;
  arma::eig_sym(vals0, H0);
  arma::eig_sym(vals1, H_final);
  EXPECT_TRUE(arma::norm(vals0 - vals1, 2) < 1e-6);
}

TEST(block_wegner_flow_reduces_off_block_coupling) {
  arma::cx_mat H0(4, 4, arma::fill::zeros);
  H0(0, 0) = 1.0;
  H0(1, 1) = 1.5;
  H0(2, 2) = 3.0;
  H0(3, 3) = 3.5;
  H0(0, 2) = 0.3;
  H0(2, 0) = 0.3;
  H0(1, 3) = 0.4;
  H0(3, 1) = 0.4;
  H0(0, 1) = 0.2;
  H0(1, 0) = 0.2;

  const size_t p_dim = 2;
  const double off0 = off_block_norm(H0, p_dim);
  const arma::cx_mat H_final =
      block_wegner_flow(H0, p_dim, 5.0, 0.01, IntegratorMethod::kRungeKutta4);
  const double off1 = off_block_norm(H_final, p_dim);

  EXPECT_TRUE(off1 < off0);
  EXPECT_TRUE(off1 < 1e-4);

  arma::vec vals0;
  arma::vec vals1;
  arma::eig_sym(vals0, H0);
  arma::eig_sym(vals1, H_final);
  EXPECT_TRUE(arma::norm(vals0 - vals1, 2) < 1e-6);
}
