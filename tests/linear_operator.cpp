#include "linear_operator.h"

#include <armadillo>

#include "framework.h"

struct DiagonalOperator final : LinearOperator<arma::vec> {
  using VectorType = arma::vec;
  using ScalarType = double;

  explicit DiagonalOperator(arma::vec diag_in) : diag(std::move(diag_in)) {}

  VectorType apply(const VectorType& v) const override { return diag % v; }
  size_t dimension() const override { return static_cast<size_t>(diag.n_elem); }

  arma::vec diag;
};

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
