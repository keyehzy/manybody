#include <armadillo>
#include <cmath>

#include "framework.h"
#include "numerics/lanczos.h"

namespace test_lanczos {
std::vector<double> solve_tridiagonal_system(const std::vector<double>& alphas,
                                             const std::vector<double>& betas) {
  const size_t n = alphas.size();
  if (n == 0) {
    return {};
  }

  arma::mat T(n, n, arma::fill::zeros);
  for (size_t i = 0; i < n; ++i) {
    T(i, i) = alphas[i];
    if (i + 1 < n) {
      T(i, i + 1) = betas[i];
      T(i + 1, i) = betas[i];
    }
  }

  arma::vec e1(n, arma::fill::zeros);
  e1(0) = 1.0;
  arma::vec y = arma::solve(T, e1);
  return std::vector<double>(y.begin(), y.end());
}

TEST(lanczos_solve_matches_direct_solution) {
  arma::mat A(5, 5, arma::fill::zeros);
  A.diag().fill(4.0);
  A(0, 1) = -1.0;
  A(1, 0) = -1.0;
  A(1, 2) = -1.0;
  A(2, 1) = -1.0;
  A(2, 3) = -1.0;
  A(3, 2) = -1.0;
  A(3, 4) = -1.0;
  A(4, 3) = -1.0;

  arma::vec b = arma::linspace<arma::vec>(1.0, 5.0, 5);

  MatrixOperator op(A);
  const size_t k = static_cast<size_t>(A.n_rows);

  arma::vec x = solve(op, b, k, solve_tridiagonal_system);
  arma::vec exact = arma::solve(A, b);

  const double rel_error = arma::norm(x - exact) / arma::norm(exact);
  EXPECT_TRUE(rel_error < 1e-8);
}

TEST(lanczos_max_eigenpair_matches_diagonal) {
  DiagonalOperator op(arma::vec{1.0, 3.0, -2.0});

  const auto eigenpair = find_max_eigenpair(op, 3);
  EXPECT_TRUE(std::abs(eigenpair.value - 3.0) < 1e-6);

  const auto applied = op.apply(eigenpair.vector);
  const double rayleigh = std::real(arma::dot(eigenpair.vector, applied)) /
                          std::real(arma::dot(eigenpair.vector, eigenpair.vector));
  EXPECT_TRUE(std::abs(rayleigh - eigenpair.value) < 1e-6);
}

TEST(lanczos_min_eigenpair_matches_diagonal) {
  DiagonalOperator op(arma::vec{1.0, 3.0, -2.0});

  const auto eigenpair = find_min_eigenpair(op, 3);
  EXPECT_TRUE(std::abs(eigenpair.value + 2.0) < 1e-6);
}

std::vector<double> exp_tridiagonal_times_e1(const std::vector<double>& alphas,
                                             const std::vector<double>& betas) {
  const size_t n = alphas.size();
  if (n == 0) {
    return {};
  }

  arma::mat T(n, n, arma::fill::zeros);
  for (size_t i = 0; i < n; ++i) {
    T(i, i) = alphas[i];
    if (i + 1 < n) {
      T(i, i + 1) = betas[i];
      T(i + 1, i) = betas[i];
    }
  }

  arma::vec eigvals;
  arma::mat eigvecs;
  if (!arma::eig_sym(eigvals, eigvecs, T)) {
    throw std::runtime_error("Eigenvalue decomposition failed for exp solver");
  }

  arma::vec exp_eigvals = arma::exp(eigvals);
  arma::mat exp_T = eigvecs * arma::diagmat(exp_eigvals) * eigvecs.t();

  arma::vec e1(n, arma::fill::zeros);
  e1(0) = 1.0;
  arma::vec y = exp_T * e1;
  return std::vector<double>(y.begin(), y.end());
}

TEST(lanczos_exp_solver_matches_direct_expm) {
  arma::mat A(4, 4, arma::fill::zeros);
  A(0, 0) = 2.0;
  A(1, 1) = 2.0;
  A(2, 2) = 2.0;
  A(3, 3) = 2.0;
  A(0, 1) = -1.0;
  A(1, 0) = -1.0;
  A(1, 2) = -1.0;
  A(2, 1) = -1.0;
  A(2, 3) = -1.0;
  A(3, 2) = -1.0;

  arma::vec b(4, arma::fill::zeros);
  b(0) = 1.0;

  MatrixOperator op(A);
  const size_t k = static_cast<size_t>(A.n_rows);

  arma::vec result = solve(op, b, k, exp_tridiagonal_times_e1);

  arma::vec eigvals;
  arma::mat eigvecs;
  EXPECT_TRUE(arma::eig_sym(eigvals, eigvecs, A));
  arma::vec exp_eigvals = arma::exp(eigvals);
  arma::mat exp_A = eigvecs * arma::diagmat(exp_eigvals) * eigvecs.t();
  arma::vec exact = exp_A * b;

  const double rel_error = arma::norm(result - exact) / arma::norm(exact);
  EXPECT_TRUE(rel_error < 1e-10);
}

TEST(lanczos_inverse_approximation_improves_with_k) {
  arma::mat A(4, 4, arma::fill::zeros);
  A(0, 0) = 10.0;
  A(1, 1) = 8.0;
  A(2, 2) = 6.0;
  A(3, 3) = 5.0;
  A(0, 1) = 1.0;
  A(1, 0) = 1.0;
  A(1, 2) = 1.0;
  A(2, 1) = 1.0;
  A(2, 3) = 1.0;
  A(3, 2) = 1.0;

  arma::vec b(4, arma::fill::ones);
  arma::vec exact = arma::solve(A, b);
  MatrixOperator op(A);

  const arma::vec x2 = solve(op, b, 2, solve_tridiagonal_system);
  const arma::vec x3 = solve(op, b, 3, solve_tridiagonal_system);
  const arma::vec x4 = solve(op, b, 4, solve_tridiagonal_system);

  const double err2 = arma::norm(x2 - exact) / arma::norm(exact);
  const double err3 = arma::norm(x3 - exact) / arma::norm(exact);
  const double err4 = arma::norm(x4 - exact) / arma::norm(exact);

  EXPECT_TRUE(err3 <= err2);
  EXPECT_TRUE(err4 <= err3);
  EXPECT_TRUE(err4 < 1e-10);
}

TEST(lanczos_decomposition_tracks_steps_and_norm) {
  arma::mat A(6, 6, arma::fill::zeros);
  A.diag().fill(4.0);
  for (size_t i = 0; i + 1 < A.n_rows; ++i) {
    A(i, i + 1) = 1.0;
    A(i + 1, i) = 1.0;
  }

  arma::vec b(6, arma::fill::ones);
  MatrixOperator op(A);
  const size_t k = 5;

  auto decomp = lanczos_pass_one(op, b, k);
  EXPECT_TRUE(decomp.steps_taken > 0);
  EXPECT_TRUE(decomp.steps_taken <= k);
  EXPECT_TRUE(decomp.alphas.size() == decomp.steps_taken);
  EXPECT_TRUE(decomp.betas.size() + 1 == decomp.steps_taken);
  EXPECT_TRUE(std::abs(decomp.b_norm - arma::norm(b)) < 1e-12);
  for (double beta : decomp.betas) {
    EXPECT_TRUE(beta > 0.0);
  }
}

struct NearestNeighbor1D final : LinearOperator<arma::vec> {
  using VectorType = arma::vec;
  using ScalarType = double;

  NearestNeighbor1D(int n, double diag_val, double hopping_val)
      : n_(n), diag_(diag_val), hopping_(hopping_val) {}

  VectorType apply(const VectorType& v) const override {
    arma::vec w(n_, arma::fill::zeros);
    if (n_ > 0) {
      w(0) = diag_ * v(0);
      if (n_ > 1) {
        w(0) += hopping_ * v(1);
      }
    }
    for (int i = 1; i < n_ - 1; ++i) {
      w(i) = hopping_ * v(i - 1) + diag_ * v(i) + hopping_ * v(i + 1);
    }
    if (n_ > 1) {
      w(n_ - 1) = hopping_ * v(n_ - 2) + diag_ * v(n_ - 1);
    }
    return w;
  }

  size_t dimension() const override { return static_cast<size_t>(n_); }

 private:
  int n_;
  double diag_;
  double hopping_;
};

TEST(lanczos_max_eigenpair_matches_nearest_neighbor_chain) {
  const int n = 80;
  const double diag = 2.0;
  const double hopping = -1.0;
  const double pi = std::acos(-1.0);

  NearestNeighbor1D op(n, diag, hopping);
  const size_t k = 50;

  auto eigenpair = find_max_eigenpair(op, k);

  const double exact_max = diag + 2.0 * hopping * std::cos(n * pi / (n + 1.0));
  const double error = std::abs(eigenpair.value - exact_max);
  EXPECT_TRUE(error < 2e-2);
}
}  // namespace test_lanczos
