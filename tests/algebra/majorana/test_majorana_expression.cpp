#include <catch2/catch.hpp>

#include "algebra/majorana/majorana_expression.h"
#include "utils/tolerances.h"

TEST_CASE("majorana_expression_construct_from_scalar") {
  MajoranaExpression expr(MajoranaExpression::complex_type(3.0, -1.0));
  CHECK(expr.size() == 1u);
  MajoranaString empty{};
  auto it = expr.hashmap.find(empty);
  CHECK(it != expr.hashmap.end());
  CHECK(it->second == MajoranaExpression::complex_type(3.0, -1.0));
}

TEST_CASE("majorana_expression_construct_from_string") {
  MajoranaString str{1, 3, 5};
  MajoranaExpression expr(MajoranaExpression::complex_type(2.0, 0.0), str);
  CHECK(expr.size() == 1u);
  auto it = expr.hashmap.find(str);
  CHECK(it != expr.hashmap.end());
  CHECK(it->second == MajoranaExpression::complex_type(2.0, 0.0));
}

TEST_CASE("majorana_expression_construct_from_sign") {
  MajoranaString str{0, 1};
  MajoranaExpression expr(-1, str);
  CHECK(expr.size() == 1u);
  auto it = expr.hashmap.find(str);
  CHECK(it != expr.hashmap.end());
  CHECK(it->second == MajoranaExpression::complex_type(-1.0, 0.0));
}

TEST_CASE("majorana_expression_add_combines_coefficients") {
  MajoranaString str{2, 4};
  MajoranaExpression a(MajoranaExpression::complex_type(1.0, 0.0), str);
  MajoranaExpression b(MajoranaExpression::complex_type(2.5, 0.0), str);

  a += b;

  auto it = a.hashmap.find(str);
  CHECK(it != a.hashmap.end());
  CHECK(it->second == MajoranaExpression::complex_type(3.5, 0.0));
  CHECK(a.size() == 1u);
}

TEST_CASE("majorana_expression_subtract_cancels") {
  MajoranaString str{1};
  MajoranaExpression a(MajoranaExpression::complex_type(1.0, 0.0), str);
  MajoranaExpression b(MajoranaExpression::complex_type(1.0, 0.0), str);

  a -= b;

  CHECK(a.size() == 0u);
}

TEST_CASE("majorana_expression_scalar_multiply") {
  MajoranaString str{0, 3};
  MajoranaExpression expr(MajoranaExpression::complex_type(2.0, 1.0), str);

  expr *= MajoranaExpression::complex_type(0.0, 1.0);

  auto it = expr.hashmap.find(str);
  CHECK(it != expr.hashmap.end());
  // (2 + i) * i = 2i + i^2 = -1 + 2i
  CHECK(std::abs(it->second - MajoranaExpression::complex_type(-1.0, 2.0)) < 1e-12);
}

TEST_CASE("majorana_expression_multiply_expressions") {
  MajoranaString str_a{0};
  MajoranaString str_b{1};
  MajoranaExpression a(MajoranaExpression::complex_type(1.0, 0.0), str_a);
  MajoranaExpression b(MajoranaExpression::complex_type(1.0, 0.0), str_b);

  a *= b;

  MajoranaString expected{0, 1};
  CHECK(a.size() == 1u);
  auto it = a.hashmap.find(expected);
  CHECK(it != a.hashmap.end());
  CHECK(it->second == MajoranaExpression::complex_type(1.0, 0.0));
}

TEST_CASE("majorana_expression_multiply_cancellation") {
  // gamma_0 * gamma_0 = identity (empty string)
  MajoranaString str{0};
  MajoranaExpression a(MajoranaExpression::complex_type(1.0, 0.0), str);
  MajoranaExpression b(MajoranaExpression::complex_type(1.0, 0.0), str);

  a *= b;

  MajoranaString empty{};
  CHECK(a.size() == 1u);
  auto it = a.hashmap.find(empty);
  CHECK(it != a.hashmap.end());
  CHECK(it->second == MajoranaExpression::complex_type(1.0, 0.0));
}

TEST_CASE("majorana_expression_norm_squared") {
  MajoranaString str_a{0, 1};
  MajoranaString str_b{2, 3};
  MajoranaExpression expr;
  expr += MajoranaExpression(MajoranaExpression::complex_type(3.0, 0.0), str_a);
  expr += MajoranaExpression(MajoranaExpression::complex_type(0.0, 4.0), str_b);

  // |3|^2 + |4i|^2 = 9 + 16 = 25
  CHECK(std::abs(expr.norm_squared() - 25.0) < 1e-12);
}

TEST_CASE("majorana_expression_zero_scalar_clears") {
  MajoranaString str{1, 2, 3};
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
