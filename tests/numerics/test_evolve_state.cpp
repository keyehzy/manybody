#include <armadillo>
#include <complex>
#include <stdexcept>
#include <vector>

#include "catch.hpp"
#include "numerics/evolve_state.h"
#include "numerics/linear_operator.h"

namespace {
struct ComplexMatrixOperator final : LinearOperator<arma::cx_vec> {
  using VectorType = arma::cx_vec;
  using ScalarType = arma::cx_double;

  explicit ComplexMatrixOperator(arma::cx_mat matrix_in) : matrix(std::move(matrix_in)) {}

  VectorType apply(const VectorType& v) const override { return matrix * v; }
  size_t dimension() const override { return static_cast<size_t>(matrix.n_rows); }

  arma::cx_mat matrix;
};

arma::cx_vec exact_time_evolution(const arma::cx_mat& H, const arma::cx_vec& psi0, double t) {
  arma::vec eigenvalues;
  arma::cx_mat eigenvectors;
  if (!arma::eig_sym(eigenvalues, eigenvectors, H)) {
    throw std::runtime_error("Eigenvalue decomposition failed for exact evolution");
  }

  arma::cx_vec phases(eigenvalues.n_elem);
  for (size_t i = 0; i < eigenvalues.n_elem; ++i) {
    phases(i) = std::exp(arma::cx_double(0.0, -t * eigenvalues(i)));
  }
  const arma::cx_mat exp_H = eigenvectors * arma::diagmat(phases) * eigenvectors.t();
  return exp_H * psi0;
}

arma::cx_vec exact_imaginary_time_evolution(const arma::cx_mat& H, const arma::cx_vec& psi0,
                                            double t) {
  arma::vec eigenvalues;
  arma::cx_mat eigenvectors;
  if (!arma::eig_sym(eigenvalues, eigenvectors, H)) {
    throw std::runtime_error("Eigenvalue decomposition failed for exact evolution");
  }

  arma::cx_vec factors(eigenvalues.n_elem);
  for (size_t i = 0; i < eigenvalues.n_elem; ++i) {
    factors(i) = std::exp(-t * eigenvalues(i));
  }
  const arma::cx_mat exp_H = eigenvectors * arma::diagmat(factors) * eigenvectors.t();
  return exp_H * psi0;
}
}  // namespace

TEST_CASE("time_evolve_state_steps_matches_exact_expm") {
  arma::vec diag = {1.0, 2.0};
  arma::cx_mat H(2, 2, arma::fill::zeros);
  H(0, 0) = diag(0);
  H(1, 1) = diag(1);

  arma::cx_vec psi0(2);
  psi0(0) = arma::cx_double(1.0, 0.0);
  psi0(1) = arma::cx_double(0.0, 1.0);

  const double t0 = 0.0;
  const double t1 = 0.3;
  const double dt = 0.1;
  EvolutionOptions<arma::cx_double> opts;
  opts.krylov_steps = 2;

  ComplexMatrixOperator op(H);
  const auto evolved = time_evolve_state_steps(op, psi0, t0, t1, dt, opts);
  const auto expected = exact_time_evolution(H, psi0, t1 - t0);

  const double rel_error = arma::norm(evolved - expected) / arma::norm(expected);
  CHECK(rel_error < 1e-8);
}

TEST_CASE("time_evolve_state_steps_calls_callback_and_matches_exact_expm") {
  arma::vec diag = {0.5, 1.5};
  arma::cx_mat H(2, 2, arma::fill::zeros);
  H(0, 0) = diag(0);
  H(1, 1) = diag(1);

  arma::cx_vec psi0(2);
  psi0(0) = arma::cx_double(0.3, -0.4);
  psi0(1) = arma::cx_double(0.2, 0.1);

  EvolutionOptions<arma::cx_double> opts;
  opts.krylov_steps = 2;

  ComplexMatrixOperator op(H);

  std::vector<double> times;
  auto callback = [&times](double time, const arma::cx_vec&) { times.push_back(time); };

  const double t0 = 0.0;
  const double t1 = 0.3;
  const double dt = 0.1;

  const auto stepped = time_evolve_state_steps(op, psi0, t0, t1, dt, opts, callback);
  const auto direct = exact_time_evolution(H, psi0, t1 - t0);

  CHECK((times.size()) == (4u));
  CHECK(std::abs(times.front() - t0) < 1e-12);
  CHECK(std::abs(times.back() - t1) < 1e-12);

  const double rel_error = arma::norm(stepped - direct) / arma::norm(direct);
  CHECK(rel_error < 1e-8);
}

TEST_CASE("imaginary_time_evolve_state_matches_exact_expm") {
  arma::vec diag = {0.2, 1.1};
  arma::cx_mat H(2, 2, arma::fill::zeros);
  H(0, 0) = diag(0);
  H(1, 1) = diag(1);

  arma::cx_vec psi0(2);
  psi0(0) = arma::cx_double(0.4, -0.1);
  psi0(1) = arma::cx_double(-0.2, 0.3);

  EvolutionOptions<arma::cx_double> opts;
  opts.krylov_steps = 2;

  ComplexMatrixOperator op(H);

  const double t = 0.7;
  const auto evolved = imaginary_time_evolve_state(op, psi0, t, opts);
  const auto expected = exact_imaginary_time_evolution(H, psi0, t);

  const double rel_error = arma::norm(evolved - expected) / arma::norm(expected);
  CHECK(rel_error < 1e-8);
}
