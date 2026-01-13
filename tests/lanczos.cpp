#include "lanczos.h"

#include <armadillo>
#include <cmath>

#include "framework.h"

namespace test_lanczos {
struct MatrixOperator final : LinearOperator<arma::vec> {
  using VectorType = arma::vec;
  using ScalarType = double;

  explicit MatrixOperator(arma::mat matrix_in) : matrix(std::move(matrix_in)) {}

  VectorType apply(const VectorType& v) const override { return matrix * v; }
  size_t dimension() const override { return static_cast<size_t>(matrix.n_rows); }

  arma::mat matrix;
};

struct DiagonalOperator final : LinearOperator<arma::vec> {
  using VectorType = arma::vec;
  using ScalarType = double;

  explicit DiagonalOperator(arma::vec diag_in) : diag(std::move(diag_in)) {}

  VectorType apply(const VectorType& v) const override { return diag % v; }
  size_t dimension() const override { return static_cast<size_t>(diag.n_elem); }

  arma::vec diag;
};

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
}  // namespace test_lanczos
