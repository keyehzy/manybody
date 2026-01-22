#include <catch2/catch.hpp>

#include "algebra/expression.h"
#include "utils/tolerances.h"

TEST_CASE("expression_construct_from_complex_identity") {
  Expression expr(Expression::complex_type(2.0f, -1.0f));
  CHECK((expr.size()) == (1u));
  Expression::container_type empty{};
  auto it = expr.hashmap.find(empty);
  CHECK(it != expr.hashmap.end());
  CHECK((it->second) == (Expression::complex_type(2.0f, -1.0f)));
}

TEST_CASE("expression_construct_from_operator") {
  Operator op = Operator::creation(Operator::Spin::Up, 3);
  Expression expr(op);
  Expression::container_type ops{op};
  auto it = expr.hashmap.find(ops);
  CHECK(it != expr.hashmap.end());
  CHECK((it->second) == (Expression::complex_type(1.0f, 0.0f)));
}

TEST_CASE("expression_initializer_list_combines_terms") {
  Operator op = Operator::annihilation(Operator::Spin::Down, 2);
  Term term_a(Expression::complex_type(2.0f, 0.0f), {op});
  Term term_b(Expression::complex_type(-2.0f, 0.0f), {op});
  Expression expr({term_a, term_b});
  CHECK((expr.size()) == (0u));
}

TEST_CASE("expression_adjoint_conjugates_and_reverses") {
  Operator a = Operator::creation(Operator::Spin::Up, 1);
  Operator b = Operator::annihilation(Operator::Spin::Down, 4);
  Term term(Expression::complex_type(1.0f, 2.0f), {a, b});
  Expression expr(term);

  Expression adj = expr.adjoint();
  Expression::container_type ops{b.adjoint(), a.adjoint()};
  auto it = adj.hashmap.find(ops);
  CHECK(it != adj.hashmap.end());
  CHECK((it->second) == (Expression::complex_type(1.0f, -2.0f)));
}

TEST_CASE("expression_add_expression_combines_coefficients") {
  Operator op = Operator::creation(Operator::Spin::Down, 5);
  Expression expr(Term(Expression::complex_type(1.0f, 0.0f), {op}));
  Expression add(Term(Expression::complex_type(2.5f, 0.0f), {op}));

  expr += add;

  Expression::container_type ops{op};
  auto it = expr.hashmap.find(ops);
  CHECK(it != expr.hashmap.end());
  CHECK((it->second) == (Expression::complex_type(3.5f, 0.0f)));
}

TEST_CASE("expression_multiply_expression_distributes") {
  Operator a = Operator::creation(Operator::Spin::Up, 1);
  Operator b = Operator::annihilation(Operator::Spin::Up, 2);
  Expression left(Term(Expression::complex_type(2.0f, 0.0f), {a}));
  Expression right(Term(Expression::complex_type(3.0f, 0.0f), {b}));

  left *= right;

  Expression::container_type ops{a, b};
  auto it = left.hashmap.find(ops);
  CHECK(it != left.hashmap.end());
  CHECK((it->second) == (Expression::complex_type(6.0f, 0.0f)));
}

TEST_CASE("expression_multiply_term_appends_ops") {
  Operator a = Operator::creation(Operator::Spin::Up, 1);
  Operator b = Operator::annihilation(Operator::Spin::Down, 7);
  Expression expr(Term(Expression::complex_type(2.0f, 0.0f), {a}));
  Term term(Expression::complex_type(0.5f, 0.0f), {b});

  expr *= term;

  Expression::container_type ops{a, b};
  auto it = expr.hashmap.find(ops);
  CHECK(it != expr.hashmap.end());
  CHECK((it->second) == (Expression::complex_type(1.0f, 0.0f)));
}

TEST_CASE("expression_ignores_near_zero_coefficients") {
  constexpr auto tolerance = tolerances::tolerance<Expression::complex_type::value_type>();
  auto small = Expression::complex_type(0.5f * tolerance, 0.0f);
  Expression expr(small);
  CHECK((expr.size()) == (0u));
}

TEST_CASE("expression_cancels_terms_within_tolerance") {
  constexpr auto tolerance = tolerances::tolerance<Expression::complex_type::value_type>();
  Operator op = Operator::creation(Operator::Spin::Up, 2);
  Expression expr(Term(Expression::complex_type(1.0f, 0.0f), {op}));
  expr += Term(Expression::complex_type(-1.0f + 0.5f * tolerance, 0.0f), {op});
  CHECK((expr.size()) == (0u));
}

TEST_CASE("expression_truncate_by_size_drops_longer_terms") {
  Operator a = Operator::creation(Operator::Spin::Up, 1);
  Operator b = Operator::creation(Operator::Spin::Up, 2);
  Operator c = Operator::annihilation(Operator::Spin::Down, 3);
  Term term_a(Expression::complex_type(1.0f, 0.0f), {a});
  Term term_b(Expression::complex_type(2.0f, 0.0f), {a, b});
  Term term_c(Expression::complex_type(3.0f, 0.0f), {a, b, c});
  Expression expr({term_a, term_b, term_c});

  expr.truncate_by_size(2);

  CHECK((expr.size()) == (2u));
  CHECK(expr.hashmap.find(Expression::container_type{a}) != expr.hashmap.end());
  CHECK(expr.hashmap.find(Expression::container_type{a, b}) != expr.hashmap.end());
  CHECK(expr.hashmap.find(Expression::container_type{a, b, c}) == expr.hashmap.end());
}

TEST_CASE("expression_truncate_by_norm_drops_small_terms") {
  Operator a = Operator::creation(Operator::Spin::Up, 1);
  Operator b = Operator::creation(Operator::Spin::Up, 2);
  Operator c = Operator::annihilation(Operator::Spin::Down, 3);
  Term term_a(Expression::complex_type(0.25f, 0.0f), {a});
  Term term_b(Expression::complex_type(0.6f, 0.0f), {b});
  Term term_c(Expression::complex_type(1.2f, 0.0f), {c});
  Expression expr({term_a, term_b, term_c});

  expr.truncate_by_norm(0.75f);

  CHECK((expr.size()) == (1u));
  CHECK(expr.hashmap.find(Expression::container_type{c}) != expr.hashmap.end());
  CHECK(expr.hashmap.find(Expression::container_type{a}) == expr.hashmap.end());
  CHECK(expr.hashmap.find(Expression::container_type{b}) == expr.hashmap.end());
}

TEST_CASE("expression_filter_by_size_keeps_exact_matches") {
  Operator a = Operator::creation(Operator::Spin::Up, 1);
  Operator b = Operator::creation(Operator::Spin::Up, 2);
  Operator c = Operator::annihilation(Operator::Spin::Down, 3);
  Term term_a(Expression::complex_type(1.0f, 0.0f), {a});
  Term term_b(Expression::complex_type(2.0f, 0.0f), {a, b});
  Term term_c(Expression::complex_type(3.0f, 0.0f), {a, b, c});
  Expression expr({term_a, term_b, term_c});

  expr.filter_by_size(2);

  CHECK((expr.size()) == (1u));
  CHECK(expr.hashmap.find(Expression::container_type{a, b}) != expr.hashmap.end());
  CHECK(expr.hashmap.find(Expression::container_type{a}) == expr.hashmap.end());
  CHECK(expr.hashmap.find(Expression::container_type{a, b, c}) == expr.hashmap.end());
}

TEST_CASE("expression_filter_by_size_zero_clears_expression") {
  Operator a = Operator::creation(Operator::Spin::Up, 1);
  Operator b = Operator::annihilation(Operator::Spin::Down, 2);
  Expression expr({Term(Expression::complex_type(1.0f, 0.0f), {a, b})});

  expr.filter_by_size(0);

  CHECK((expr.size()) == (0u));
}
