#include <catch2/catch.hpp>

#include "algebra/fermion/expression.h"

TEST_CASE("commutator_commuting_creations_doubles_ordered_term") {
  FermionOperator a = FermionOperator::creation(FermionOperator::Spin::Up, 1);
  FermionOperator b = FermionOperator::creation(FermionOperator::Spin::Up, 2);
  FermionExpression left(a);
  FermionExpression right(b);

  FermionExpression result = commutator(left, right);

  FermionExpression::container_type ordered{a, b};
  auto it = result.terms().find(ordered);
  CHECK(it != result.terms().end());
  CHECK((it->second) == (FermionExpression::complex_type(2.0f, 0.0f)));
  CHECK((result.size()) == (1u));
}

TEST_CASE("anticommutator_commuting_creations_vanishes") {
  FermionOperator a = FermionOperator::creation(FermionOperator::Spin::Up, 1);
  FermionOperator b = FermionOperator::creation(FermionOperator::Spin::Up, 2);
  FermionExpression left(a);
  FermionExpression right(b);

  FermionExpression result = anticommutator(left, right);
  CHECK((result.size()) == (0u));
}

TEST_CASE("commutator_creation_annihilation_same_orbital") {
  FermionOperator create = FermionOperator::creation(FermionOperator::Spin::Down, 3);
  FermionOperator annihilate = FermionOperator::annihilation(FermionOperator::Spin::Down, 3);
  FermionExpression left(create);
  FermionExpression right(annihilate);

  FermionExpression result = commutator(left, right);

  FermionExpression::container_type empty{};
  auto it_empty = result.terms().find(empty);
  CHECK(it_empty != result.terms().end());
  CHECK((it_empty->second) == (FermionExpression::complex_type(-1.0f, 0.0f)));

  FermionExpression::container_type ordered{create, annihilate};
  auto it_ordered = result.terms().find(ordered);
  CHECK(it_ordered != result.terms().end());
  CHECK((it_ordered->second) == (FermionExpression::complex_type(2.0f, 0.0f)));
  CHECK((result.size()) == (2u));
}

TEST_CASE("anticommutator_creation_annihilation_same_orbital_is_identity") {
  FermionOperator create = FermionOperator::creation(FermionOperator::Spin::Down, 3);
  FermionOperator annihilate = FermionOperator::annihilation(FermionOperator::Spin::Down, 3);
  FermionExpression left(create);
  FermionExpression right(annihilate);

  FermionExpression result = anticommutator(left, right);

  FermionExpression::container_type empty{};
  auto it_empty = result.terms().find(empty);
  CHECK(it_empty != result.terms().end());
  CHECK((it_empty->second) == (FermionExpression::complex_type(1.0f, 0.0f)));
  CHECK((result.size()) == (1u));
}

TEST_CASE("commutator_expression_distributes_over_terms") {
  FermionOperator a = FermionOperator::creation(FermionOperator::Spin::Up, 1);
  FermionOperator b = FermionOperator::creation(FermionOperator::Spin::Up, 2);
  FermionOperator annihilate = FermionOperator::annihilation(FermionOperator::Spin::Up, 1);
  FermionExpression term_a(a);
  FermionExpression term_b(b);
  FermionExpression term_c(annihilate);

  FermionExpression left = term_a + term_b;
  FermionExpression right(term_c);

  FermionExpression result = commutator(left, right);
  FermionExpression expected = commutator(term_a, term_c);
  expected += commutator(term_b, term_c);

  CHECK((result.size()) == (expected.size()));
  for (const auto& [ops, coeff] : expected.terms()) {
    auto it = result.terms().find(ops);
    CHECK(it != result.terms().end());
    CHECK((it->second) == (coeff));
  }
}

TEST_CASE("bch_order_one_matches_first_commutator_term") {
  FermionOperator create = FermionOperator::creation(FermionOperator::Spin::Down, 3);
  FermionOperator annihilate = FermionOperator::annihilation(FermionOperator::Spin::Down, 3);
  FermionExpression A{FermionMonomial(create)};
  FermionExpression B{FermionMonomial(annihilate)};

  const FermionExpression::complex_type::value_type lambda = 0.5;
  FermionExpression result = BCH(A, B, lambda, 1);

  FermionExpression expected = B + (commutator(A, B) * lambda);
  CHECK((result.size()) == (expected.size()));
  for (const auto& [ops, coeff] : expected.terms()) {
    auto it = result.terms().find(ops);
    CHECK(it != result.terms().end());
    CHECK((it->second) == (coeff));
  }
}

TEST_CASE("bch_zero_lambda_returns_b") {
  FermionOperator create = FermionOperator::creation(FermionOperator::Spin::Up, 2);
  FermionOperator annihilate = FermionOperator::annihilation(FermionOperator::Spin::Up, 2);
  FermionExpression A{FermionMonomial(create)};
  FermionExpression B{FermionMonomial(annihilate)};

  FermionExpression result = BCH(A, B, 0.0, 5);

  CHECK((result.size()) == (B.size()));
  for (const auto& [ops, coeff] : B.terms()) {
    auto it = result.terms().find(ops);
    CHECK(it != result.terms().end());
    CHECK((it->second) == (coeff));
  }
}

TEST_CASE("bch_identity_generator_leaves_b") {
  FermionOperator create = FermionOperator::creation(FermionOperator::Spin::Down, 1);
  FermionExpression A(FermionExpression::complex_type(1.0f, 0.0f));
  FermionExpression B{FermionMonomial(create)};

  FermionExpression result = BCH(A, B, 0.75, 4);

  CHECK((result.size()) == (B.size()));
  for (const auto& [ops, coeff] : B.terms()) {
    auto it = result.terms().find(ops);
    CHECK(it != result.terms().end());
    CHECK((it->second) == (coeff));
  }
}
