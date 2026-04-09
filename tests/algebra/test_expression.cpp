#include <catch2/catch.hpp>

#include "algebra/fermion/expression.h"
#include "utils/tolerances.h"

TEST_CASE("expression_construct_from_complex_identity") {
  FermionExpression expr(FermionExpression::complex_type(2.0f, -1.0f));
  CHECK((expr.size()) == (1u));
  FermionExpression::container_type empty{};
  auto it = expr.terms().find(empty);
  CHECK(it != expr.terms().end());
  CHECK((it->second) == (FermionExpression::complex_type(2.0f, -1.0f)));
}

TEST_CASE("expression_construct_from_operator") {
  FermionOperator op = FermionOperator::creation(FermionOperator::Spin::Up, 3);
  FermionExpression expr(op);
  FermionExpression::container_type ops{op};
  auto it = expr.terms().find(ops);
  CHECK(it != expr.terms().end());
  CHECK((it->second) == (FermionExpression::complex_type(1.0f, 0.0f)));
}

TEST_CASE("expression_initializer_list_combines_terms") {
  FermionOperator op = FermionOperator::annihilation(FermionOperator::Spin::Down, 2);
  FermionMonomial term_a(FermionExpression::complex_type(2.0f, 0.0f), {op});
  FermionMonomial term_b(FermionExpression::complex_type(-2.0f, 0.0f), {op});
  FermionExpression expr({term_a, term_b});
  CHECK((expr.size()) == (0u));
}

TEST_CASE("expression_adjoint_conjugates_and_reverses") {
  FermionOperator a = FermionOperator::creation(FermionOperator::Spin::Up, 1);
  FermionOperator b = FermionOperator::annihilation(FermionOperator::Spin::Down, 4);
  FermionMonomial term(FermionExpression::complex_type(1.0f, 2.0f), {a, b});
  FermionExpression expr(term);

  FermionExpression adj = adjoint(expr);
  FermionExpression::container_type ops{b.adjoint(), a.adjoint()};
  auto it = adj.terms().find(ops);
  CHECK(it != adj.terms().end());
  CHECK((it->second) == (FermionExpression::complex_type(1.0f, -2.0f)));
}

TEST_CASE("expression_add_expression_combines_coefficients") {
  FermionOperator op = FermionOperator::creation(FermionOperator::Spin::Down, 5);
  FermionExpression expr(FermionMonomial(FermionExpression::complex_type(1.0f, 0.0f), {op}));
  FermionExpression add(FermionMonomial(FermionExpression::complex_type(2.5f, 0.0f), {op}));

  expr += add;

  FermionExpression::container_type ops{op};
  auto it = expr.terms().find(ops);
  CHECK(it != expr.terms().end());
  CHECK((it->second) == (FermionExpression::complex_type(3.5f, 0.0f)));
}

TEST_CASE("expression_multiply_expression_distributes") {
  FermionOperator a = FermionOperator::creation(FermionOperator::Spin::Up, 1);
  FermionOperator b = FermionOperator::annihilation(FermionOperator::Spin::Up, 2);
  FermionExpression left(FermionMonomial(FermionExpression::complex_type(2.0f, 0.0f), {a}));
  FermionExpression right(FermionMonomial(FermionExpression::complex_type(3.0f, 0.0f), {b}));

  left *= right;

  FermionExpression::container_type ops{a, b};
  auto it = left.terms().find(ops);
  CHECK(it != left.terms().end());
  CHECK((it->second) == (FermionExpression::complex_type(6.0f, 0.0f)));
}

TEST_CASE("expression_multiply_term_appends_ops") {
  FermionOperator a = FermionOperator::creation(FermionOperator::Spin::Up, 1);
  FermionOperator b = FermionOperator::annihilation(FermionOperator::Spin::Down, 7);
  FermionExpression expr(FermionMonomial(FermionExpression::complex_type(2.0f, 0.0f), {a}));
  FermionMonomial term(FermionExpression::complex_type(0.5f, 0.0f), {b});

  expr *= term;

  FermionExpression::container_type ops{a, b};
  auto it = expr.terms().find(ops);
  CHECK(it != expr.terms().end());
  CHECK((it->second) == (FermionExpression::complex_type(1.0f, 0.0f)));
}

TEST_CASE("expression_ignores_near_zero_coefficients") {
  constexpr auto tolerance = tolerances::tolerance<FermionExpression::complex_type::value_type>();
  auto small = FermionExpression::complex_type(0.5f * tolerance, 0.0f);
  FermionExpression expr(small);
  CHECK((expr.size()) == (0u));
}

TEST_CASE("expression_cancels_terms_within_tolerance") {
  constexpr auto tolerance = tolerances::tolerance<FermionExpression::complex_type::value_type>();
  FermionOperator op = FermionOperator::creation(FermionOperator::Spin::Up, 2);
  FermionExpression expr(FermionMonomial(FermionExpression::complex_type(1.0f, 0.0f), {op}));
  expr += FermionMonomial(FermionExpression::complex_type(-1.0f + 0.5f * tolerance, 0.0f), {op});
  CHECK((expr.size()) == (0u));
}

TEST_CASE("expression_truncate_by_size_drops_longer_terms") {
  FermionOperator a = FermionOperator::creation(FermionOperator::Spin::Up, 1);
  FermionOperator b = FermionOperator::creation(FermionOperator::Spin::Up, 2);
  FermionOperator c = FermionOperator::annihilation(FermionOperator::Spin::Down, 3);
  FermionMonomial term_a(FermionExpression::complex_type(1.0f, 0.0f), {a});
  FermionMonomial term_b(FermionExpression::complex_type(2.0f, 0.0f), {a, b});
  FermionMonomial term_c(FermionExpression::complex_type(3.0f, 0.0f), {a, b, c});
  FermionExpression expr({term_a, term_b, term_c});

  expr.truncate_by_size(2);

  CHECK((expr.size()) == (2u));
  CHECK(expr.terms().find(FermionExpression::container_type{a}) != expr.terms().end());
  CHECK(expr.terms().find(FermionExpression::container_type{a, b}) != expr.terms().end());
  CHECK(expr.terms().find(FermionExpression::container_type{a, b, c}) == expr.terms().end());
}

TEST_CASE("expression_truncate_by_norm_drops_small_terms") {
  FermionOperator a = FermionOperator::creation(FermionOperator::Spin::Up, 1);
  FermionOperator b = FermionOperator::creation(FermionOperator::Spin::Up, 2);
  FermionOperator c = FermionOperator::annihilation(FermionOperator::Spin::Down, 3);
  FermionMonomial term_a(FermionExpression::complex_type(0.25f, 0.0f), {a});
  FermionMonomial term_b(FermionExpression::complex_type(0.6f, 0.0f), {b});
  FermionMonomial term_c(FermionExpression::complex_type(1.2f, 0.0f), {c});
  FermionExpression expr({term_a, term_b, term_c});

  expr.truncate_by_norm(0.75f);

  CHECK((expr.size()) == (1u));
  CHECK(expr.terms().find(FermionExpression::container_type{c}) != expr.terms().end());
  CHECK(expr.terms().find(FermionExpression::container_type{a}) == expr.terms().end());
  CHECK(expr.terms().find(FermionExpression::container_type{b}) == expr.terms().end());
}

TEST_CASE("expression_filter_by_size_keeps_exact_matches") {
  FermionOperator a = FermionOperator::creation(FermionOperator::Spin::Up, 1);
  FermionOperator b = FermionOperator::creation(FermionOperator::Spin::Up, 2);
  FermionOperator c = FermionOperator::annihilation(FermionOperator::Spin::Down, 3);
  FermionMonomial term_a(FermionExpression::complex_type(1.0f, 0.0f), {a});
  FermionMonomial term_b(FermionExpression::complex_type(2.0f, 0.0f), {a, b});
  FermionMonomial term_c(FermionExpression::complex_type(3.0f, 0.0f), {a, b, c});
  FermionExpression expr({term_a, term_b, term_c});

  expr.filter_by_size(2);

  CHECK((expr.size()) == (1u));
  CHECK(expr.terms().find(FermionExpression::container_type{a, b}) != expr.terms().end());
  CHECK(expr.terms().find(FermionExpression::container_type{a}) == expr.terms().end());
  CHECK(expr.terms().find(FermionExpression::container_type{a, b, c}) == expr.terms().end());
}

TEST_CASE("expression_filter_by_size_zero_clears_expression") {
  FermionOperator a = FermionOperator::creation(FermionOperator::Spin::Up, 1);
  FermionOperator b = FermionOperator::annihilation(FermionOperator::Spin::Down, 2);
  FermionExpression expr({FermionMonomial(FermionExpression::complex_type(1.0f, 0.0f), {a, b})});

  expr.filter_by_size(0);

  CHECK((expr.size()) == (0u));
}
