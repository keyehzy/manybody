#include "linear_operator.h"

#include <armadillo>
#include <cmath>

#include "exp_operator.h"
#include "framework.h"
#include "linear_operator_utils.h"

TEST(linear_operator_negated_applies_sign) {
  DiagonalOperator op(arma::vec{1.0, 2.0, -1.0});
  arma::vec v{2.0, -1.0, 3.0};

  auto neg = -op;
  arma::vec result = neg.apply(v);

  EXPECT_EQ(result(0), -2.0);
  EXPECT_EQ(result(1), 2.0);
  EXPECT_EQ(result(2), 3.0);
  EXPECT_EQ(neg.dimension(), 3u);
}

TEST(linear_operator_scaled_applies_factor) {
  DiagonalOperator op(arma::vec{1.0, -2.0, 0.5});
  arma::vec v{2.0, 3.0, -4.0};

  auto scaled = op * 2.0;
  arma::vec result = scaled.apply(v);

  EXPECT_EQ(result(0), 4.0);
  EXPECT_EQ(result(1), -12.0);
  EXPECT_EQ(result(2), -4.0);
  EXPECT_EQ(scaled.dimension(), 3u);
}

TEST(linear_operator_scaled_applies_factor_left) {
  DiagonalOperator op(arma::vec{1.0, -2.0, 0.5});
  arma::vec v{2.0, 3.0, -4.0};

  auto scaled = 2.0 * op;
  arma::vec result = scaled.apply(v);

  EXPECT_EQ(result(0), 4.0);
  EXPECT_EQ(result(1), -12.0);
  EXPECT_EQ(result(2), -4.0);
  EXPECT_EQ(scaled.dimension(), 3u);
}

TEST(linear_operator_sum_adds_results) {
  DiagonalOperator a(arma::vec{1.0, 2.0, 3.0});
  DiagonalOperator b(arma::vec{-1.0, 4.0, 0.5});
  arma::vec v{2.0, -1.0, 3.0};

  auto sum = a + b;
  arma::vec result = sum.apply(v);

  EXPECT_EQ(result(0), 0.0);
  EXPECT_EQ(result(1), -6.0);
  EXPECT_EQ(result(2), 10.5);
  EXPECT_EQ(sum.dimension(), 3u);
}

TEST(linear_operator_difference_subtracts_results) {
  DiagonalOperator a(arma::vec{1.0, 2.0, 3.0});
  DiagonalOperator b(arma::vec{-1.0, 4.0, 0.5});
  arma::vec v{2.0, -1.0, 3.0};

  auto diff = a - b;
  arma::vec result = diff.apply(v);

  EXPECT_EQ(result(0), 4.0);
  EXPECT_EQ(result(1), 2.0);
  EXPECT_EQ(result(2), 7.5);
  EXPECT_EQ(diff.dimension(), 3u);
}

TEST(linear_operator_composed_applies_in_sequence) {
  DiagonalOperator a(arma::vec{2.0, 0.5, -1.0});
  DiagonalOperator b(arma::vec{3.0, -2.0, 4.0});
  arma::vec v{1.0, -1.0, 2.0};

  auto composed = a * b;
  arma::vec result = composed.apply(v);

  EXPECT_EQ(result(0), 6.0);
  EXPECT_EQ(result(1), 1.0);
  EXPECT_EQ(result(2), -8.0);
  EXPECT_EQ(composed.dimension(), 3u);
}

TEST(linear_operator_identity_preserves_vector) {
  Identity<arma::vec> identity(3);
  arma::vec v{1.0, -2.0, 5.0};

  arma::vec result = identity.apply(v);

  EXPECT_EQ(result(0), 1.0);
  EXPECT_EQ(result(1), -2.0);
  EXPECT_EQ(result(2), 5.0);
  EXPECT_EQ(identity.dimension(), 3u);
}

TEST(linear_operator_shifted_applies_shift) {
  DiagonalOperator op(arma::vec{1.0, 2.0, -1.0});
  arma::vec v{2.0, -1.0, 3.0};

  Shifted<DiagonalOperator> shifted(op, 1.5);
  arma::vec result = shifted.apply(v);

  EXPECT_EQ(result(0), 5.0);
  EXPECT_EQ(result(1), -3.5);
  EXPECT_EQ(result(2), 1.5);
  EXPECT_EQ(shifted.dimension(), 3u);
}

TEST(linear_operator_neg_shift_applies_negated_shift) {
  DiagonalOperator op(arma::vec{1.0, 2.0, -1.0});
  arma::vec v{2.0, -1.0, 3.0};

  NegShift<DiagonalOperator> shifted(op, 1.5);
  arma::vec result = shifted.apply(v);

  EXPECT_EQ(result(0), 1.0);
  EXPECT_EQ(result(1), 0.5);
  EXPECT_EQ(result(2), 7.5);
  EXPECT_EQ(shifted.dimension(), 3u);
}

TEST(linear_operator_power_method_estimates_dominant_eigenvalue) {
  DiagonalOperator op(arma::vec{3.0, -1.0, 0.5});
  arma::vec seed{1.0, 1.0, 1.0};

  const double dominant = power_method(op, seed, 40);

  EXPECT_TRUE(std::abs(dominant - 3.0) < 1e-3);
}

TEST(linear_operator_estimate_bounds_matches_diagonal_spectrum) {
  DiagonalOperator op(arma::vec{-2.0, 0.5, 4.0});

  const auto bounds = estimate_bounds(op, 40, 0.2);

  EXPECT_TRUE(std::abs(bounds.alpha + 2.0) < 1e-3);
  EXPECT_TRUE(std::abs(bounds.beta - 4.0) < 1e-3);
}

TEST(linear_operator_exp_matches_diagonal_exponential) {
  DiagonalOperator op(arma::vec{-2.0, 0.5, 1.5});
  arma::vec v{1.0, -2.0, 0.5};

  ExpOptions<double> options;
  options.krylov_steps = op.dimension();
  Exp<DiagonalOperator> exp_op(op, options);
  arma::vec result = exp_op.apply(v);

  arma::vec expected = arma::vec{std::exp(-2.0), std::exp(0.5), std::exp(1.5)} % v;
  const double tol = 1e-6;
  for (size_t i = 0; i < expected.n_elem; ++i) {
    EXPECT_TRUE(std::abs(result(i) - expected(i)) < tol);
  }
}

TEST(linear_operator_exp_handles_scaled_steps) {
  DiagonalOperator op(arma::vec{-3.0, 1.0});
  arma::vec v{1.0, 2.0};

  ExpOptions<double> options;
  options.krylov_steps = 50;
  Exp<DiagonalOperator> exp_op(op, options);
  arma::vec result = exp_op.apply(v);

  arma::vec expected = arma::vec{std::exp(-3.0), std::exp(1.0)} % v;
  const double tol = 1e-5;
  for (size_t i = 0; i < expected.n_elem; ++i) {
    EXPECT_TRUE(std::abs(result(i) - expected(i)) < tol);
  }
}
