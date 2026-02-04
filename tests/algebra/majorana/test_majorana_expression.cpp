#include <catch2/catch.hpp>

#include "algebra/majorana/expression.h"
#include "utils/tolerances.h"

using namespace majorana;

namespace majorana_expression_tests {

static MajoranaMonomial::container_type make_string(
    std::initializer_list<MajoranaOperator> elements) {
  MajoranaMonomial::container_type str;
  str.append_range(elements.begin(), elements.end());
  return str;
}

static MajoranaOperator even(size_t orbital, Operator::Spin spin) {
  return MajoranaOperator::even(orbital, spin);
}

static MajoranaOperator odd(size_t orbital, Operator::Spin spin) {
  return MajoranaOperator::odd(orbital, spin);
}

TEST_CASE("majorana_expression_construct_from_scalar") {
  MajoranaExpression expr(MajoranaExpression::complex_type(3.0, -1.0));
  CHECK(expr.size() == 1u);
  MajoranaMonomial::container_type empty;
  auto it = expr.terms().find(empty);
  CHECK(it != expr.terms().end());
  CHECK(it->second == MajoranaExpression::complex_type(3.0, -1.0));
}

TEST_CASE("majorana_expression_construct_from_string") {
  MajoranaMonomial::container_type str = make_string(
      {even(0, Operator::Spin::Down), odd(0, Operator::Spin::Down), even(1, Operator::Spin::Down)});
  MajoranaExpression expr(MajoranaExpression::complex_type(2.0, 0.0), str);
  CHECK(expr.size() == 1u);
  auto it = expr.terms().find(str);
  CHECK(it != expr.terms().end());
  CHECK(it->second == MajoranaExpression::complex_type(2.0, 0.0));
}

TEST_CASE("majorana_expression_construct_from_sign") {
  MajoranaMonomial::container_type str =
      make_string({even(0, Operator::Spin::Up), even(0, Operator::Spin::Down)});
  MajoranaExpression expr(-1, str);
  CHECK(expr.size() == 1u);
  auto it = expr.terms().find(str);
  CHECK(it != expr.terms().end());
  CHECK(it->second == MajoranaExpression::complex_type(-1.0, 0.0));
}

TEST_CASE("majorana_expression_add_combines_coefficients") {
  MajoranaMonomial::container_type str =
      make_string({odd(0, Operator::Spin::Up), even(1, Operator::Spin::Up)});
  MajoranaExpression a(MajoranaExpression::complex_type(1.0, 0.0), str);
  MajoranaExpression b(MajoranaExpression::complex_type(2.5, 0.0), str);

  a += b;

  auto it = a.terms().find(str);
  CHECK(it != a.terms().end());
  CHECK(it->second == MajoranaExpression::complex_type(3.5, 0.0));
  CHECK(a.size() == 1u);
}

TEST_CASE("majorana_expression_subtract_cancels") {
  MajoranaMonomial::container_type str = make_string({even(0, Operator::Spin::Down)});
  MajoranaExpression a(MajoranaExpression::complex_type(1.0, 0.0), str);
  MajoranaExpression b(MajoranaExpression::complex_type(1.0, 0.0), str);

  a -= b;

  CHECK(a.size() == 0u);
}

TEST_CASE("majorana_expression_scalar_multiply") {
  MajoranaMonomial::container_type str =
      make_string({even(0, Operator::Spin::Up), odd(0, Operator::Spin::Down)});
  MajoranaExpression expr(MajoranaExpression::complex_type(2.0, 1.0), str);

  expr *= MajoranaExpression::complex_type(0.0, 1.0);

  auto it = expr.terms().find(str);
  CHECK(it != expr.terms().end());
  // (2 + i) * i = 2i + i^2 = -1 + 2i
  CHECK(std::abs(it->second - MajoranaExpression::complex_type(-1.0, 2.0)) < 1e-12);
}

TEST_CASE("majorana_expression_multiply_expressions") {
  MajoranaMonomial::container_type str_a = make_string({even(0, Operator::Spin::Up)});
  MajoranaMonomial::container_type str_b = make_string({even(0, Operator::Spin::Down)});
  MajoranaExpression a(MajoranaExpression::complex_type(1.0, 0.0), str_a);
  MajoranaExpression b(MajoranaExpression::complex_type(1.0, 0.0), str_b);

  a *= b;

  MajoranaMonomial::container_type expected =
      make_string({even(0, Operator::Spin::Up), even(0, Operator::Spin::Down)});
  CHECK(a.size() == 1u);
  auto it = a.terms().find(expected);
  CHECK(it != a.terms().end());
  CHECK(it->second == MajoranaExpression::complex_type(1.0, 0.0));
}

TEST_CASE("majorana_expression_multiply_cancellation") {
  // gamma_0 * gamma_0 = identity (empty string)
  MajoranaMonomial::container_type str = make_string({even(0, Operator::Spin::Up)});
  MajoranaExpression a(MajoranaExpression::complex_type(1.0, 0.0), str);
  MajoranaExpression b(MajoranaExpression::complex_type(1.0, 0.0), str);

  a *= b;

  MajoranaMonomial::container_type empty;
  CHECK(a.size() == 1u);
  auto it = a.terms().find(empty);
  CHECK(it != a.terms().end());
  CHECK(it->second == MajoranaExpression::complex_type(1.0, 0.0));
}

TEST_CASE("majorana_expression_zero_scalar_clears") {
  MajoranaMonomial::container_type str = make_string(
      {even(0, Operator::Spin::Down), odd(0, Operator::Spin::Up), odd(0, Operator::Spin::Down)});
  MajoranaExpression expr(MajoranaExpression::complex_type(5.0, 0.0), str);

  expr *= MajoranaExpression::complex_type(0.0, 0.0);

  CHECK(expr.size() == 0u);
}

TEST_CASE("majorana_expression_ignores_near_zero_coefficients") {
  constexpr auto tol = tolerances::tolerance<double>();
  auto small = MajoranaExpression::complex_type(0.5 * tol, 0.0);
  MajoranaExpression expr(small);
  CHECK(expr.size() == 0u);
}

}  // namespace majorana_expression_tests
