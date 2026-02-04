#include <catch2/catch.hpp>

#include "algebra/expression.h"
#include "utils/tolerances.h"

using ExpressionType = FermionExpression;
using complex_type = ExpressionType::complex_type;
using container_type = ExpressionType::container_type;

TEST_CASE("expression_is_zero_below_tolerance") {
  constexpr auto tol = tolerances::tolerance<double>();
  CHECK(ExpressionType::is_zero(complex_type(0.0, 0.0)));
  CHECK(ExpressionType::is_zero(complex_type(0.5 * tol, 0.0)));
  CHECK(ExpressionType::is_zero(complex_type(0.0, 0.5 * tol)));
  CHECK_FALSE(ExpressionType::is_zero(complex_type(1.0, 0.0)));
  CHECK_FALSE(ExpressionType::is_zero(complex_type(0.0, 1.0)));
}

TEST_CASE("expression_add_inserts_new_key") {
  ExpressionType expr;
  container_type key{Operator::creation(Operator::Spin::Up, 0)};
  expr.add(key, complex_type(1.0, 0.0));
  CHECK(expr.size() == 1u);
  auto it = expr.data.find(key);
  CHECK(it != expr.data.end());
  CHECK(it->second == complex_type(1.0, 0.0));
}

TEST_CASE("expression_add_combines_existing_key") {
  ExpressionType expr;
  container_type key{Operator::creation(Operator::Spin::Up, 0)};
  expr.add(key, complex_type(1.0, 0.0));
  expr.add(key, complex_type(2.0, 0.0));
  CHECK(expr.size() == 1u);
  CHECK(expr.data[key] == complex_type(3.0, 0.0));
}

TEST_CASE("expression_add_prunes_zero") {
  ExpressionType expr;
  container_type key{Operator::creation(Operator::Spin::Up, 0)};
  expr.add(key, complex_type(1.0, 0.0));
  expr.add(key, complex_type(-1.0, 0.0));
  CHECK(expr.size() == 0u);
}

TEST_CASE("expression_add_ignores_zero_coeff") {
  ExpressionType expr;
  container_type key{Operator::creation(Operator::Spin::Up, 0)};
  expr.add(key, complex_type(0.0, 0.0));
  CHECK(expr.size() == 0u);
}

TEST_CASE("expression_add_scalar") {
  ExpressionType expr;
  expr.add_scalar(complex_type(2.0, 1.0));
  CHECK(expr.size() == 1u);
  auto it = expr.data.find(container_type{});
  CHECK(it != expr.data.end());
  CHECK(it->second == complex_type(2.0, 1.0));
}

TEST_CASE("expression_subtract_scalar") {
  ExpressionType expr;
  expr.add_scalar(complex_type(2.0, 1.0));
  expr.subtract_scalar(complex_type(2.0, 1.0));
  CHECK(expr.size() == 0u);
}

TEST_CASE("expression_scale") {
  ExpressionType expr;
  container_type key{Operator::creation(Operator::Spin::Up, 0)};
  expr.add(key, complex_type(2.0, 0.0));
  expr.scale(complex_type(3.0, 0.0));
  CHECK(expr.data[key] == complex_type(6.0, 0.0));
}

TEST_CASE("expression_scale_by_zero_clears") {
  ExpressionType expr;
  expr.add(container_type{Operator::creation(Operator::Spin::Up, 0)}, complex_type(2.0, 0.0));
  expr.scale(complex_type(0.0, 0.0));
  CHECK(expr.empty());
}

TEST_CASE("expression_divide") {
  ExpressionType expr;
  container_type key{Operator::creation(Operator::Spin::Up, 0)};
  expr.add(key, complex_type(6.0, 0.0));
  expr.divide(complex_type(3.0, 0.0));
  CHECK(expr.data[key] == complex_type(2.0, 0.0));
}

TEST_CASE("expression_add_all") {
  ExpressionType a;
  ExpressionType b;
  container_type k1{Operator::creation(Operator::Spin::Up, 0)};
  container_type k2{Operator::creation(Operator::Spin::Down, 1)};
  a.add(k1, complex_type(1.0, 0.0));
  b.add(k1, complex_type(2.0, 0.0));
  b.add(k2, complex_type(3.0, 0.0));
  a.add_all(b);
  CHECK(a.size() == 2u);
  CHECK(a.data[k1] == complex_type(3.0, 0.0));
  CHECK(a.data[k2] == complex_type(3.0, 0.0));
}

TEST_CASE("expression_subtract_all") {
  ExpressionType a;
  ExpressionType b;
  container_type k1{Operator::creation(Operator::Spin::Up, 0)};
  a.add(k1, complex_type(3.0, 0.0));
  b.add(k1, complex_type(3.0, 0.0));
  a.subtract_all(b);
  CHECK(a.empty());
}

TEST_CASE("expression_truncate_by_norm") {
  ExpressionType expr;
  container_type k1{Operator::creation(Operator::Spin::Up, 0)};
  container_type k2{Operator::creation(Operator::Spin::Down, 1)};
  expr.add(k1, complex_type(0.1, 0.0));
  expr.add(k2, complex_type(2.0, 0.0));
  expr.truncate_by_norm(0.5);
  CHECK(expr.size() == 1u);
  CHECK(expr.data.find(k2) != expr.data.end());
}

TEST_CASE("expression_truncate_by_norm_zero_is_noop") {
  ExpressionType expr;
  expr.add(container_type{}, complex_type(0.1, 0.0));
  expr.truncate_by_norm(0.0);
  CHECK(expr.size() == 1u);
}

TEST_CASE("expression_format_sorted_empty") {
  ExpressionType expr;
  std::ostringstream oss;
  expr.format_sorted(
      oss, [](std::ostringstream& os, const container_type&, const complex_type&) { os << "x"; });
  CHECK(oss.str() == "0");
}

TEST_CASE("expression_format_sorted_orders_by_size_then_norm") {
  ExpressionType expr;
  Operator a = Operator::creation(Operator::Spin::Up, 0);
  Operator b = Operator::creation(Operator::Spin::Up, 1);
  container_type k_empty{};
  container_type k_one{a};
  container_type k_two{a, b};

  expr.add(k_two, complex_type(5.0, 0.0));
  expr.add(k_empty, complex_type(1.0, 0.0));
  expr.add(k_one, complex_type(3.0, 0.0));

  std::vector<size_t> sizes;
  std::ostringstream oss;
  expr.format_sorted(
      oss, [&sizes](std::ostringstream& os, const container_type& key, const complex_type& c) {
        sizes.push_back(key.size());
        os << std::abs(c);
      });

  REQUIRE(sizes.size() == 3u);
  CHECK(sizes[0] == 0u);
  CHECK(sizes[1] == 1u);
  CHECK(sizes[2] == 2u);
}

TEST_CASE("expression_empty_operations") {
  ExpressionType expr;
  CHECK(expr.empty());
  CHECK(expr.size() == 0u);
  expr.scale(complex_type(2.0, 0.0));
  CHECK(expr.empty());
  expr.truncate_by_norm(1.0);
  CHECK(expr.empty());
  ExpressionType other;
  expr.add_all(other);
  CHECK(expr.empty());
}
