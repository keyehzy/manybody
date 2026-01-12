#include "algorithm/wegner_flow.h"

#include <armadillo>
#include <cmath>

#include "algorithm/dynamical_system.h"
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
