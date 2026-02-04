#include <catch2/catch.hpp>

#include "algebra/commutator.h"

TEST_CASE("commutator_commuting_creations_doubles_ordered_term") {
  Operator a = Operator::creation(Operator::Spin::Up, 1);
  Operator b = Operator::creation(Operator::Spin::Up, 2);
  FermionMonomial left(a);
  FermionMonomial right(b);

  Expression result = commutator(left, right);

  Expression::container_type ordered{a, b};
  auto it = result.terms().find(ordered);
  CHECK(it != result.terms().end());
  CHECK((it->second) == (Expression::complex_type(2.0f, 0.0f)));
  CHECK((result.size()) == (1u));
}

TEST_CASE("anticommutator_commuting_creations_vanishes") {
  Operator a = Operator::creation(Operator::Spin::Up, 1);
  Operator b = Operator::creation(Operator::Spin::Up, 2);
  FermionMonomial left(a);
  FermionMonomial right(b);

  Expression result = anticommutator(left, right);
  CHECK((result.size()) == (0u));
}

TEST_CASE("commutator_creation_annihilation_same_orbital") {
  Operator create = Operator::creation(Operator::Spin::Down, 3);
  Operator annihilate = Operator::annihilation(Operator::Spin::Down, 3);
  FermionMonomial left(create);
  FermionMonomial right(annihilate);

  Expression result = commutator(left, right);

  Expression::container_type empty{};
  auto it_empty = result.terms().find(empty);
  CHECK(it_empty != result.terms().end());
  CHECK((it_empty->second) == (Expression::complex_type(-1.0f, 0.0f)));

  Expression::container_type ordered{create, annihilate};
  auto it_ordered = result.terms().find(ordered);
  CHECK(it_ordered != result.terms().end());
  CHECK((it_ordered->second) == (Expression::complex_type(2.0f, 0.0f)));
  CHECK((result.size()) == (2u));
}

TEST_CASE("anticommutator_creation_annihilation_same_orbital_is_identity") {
  Operator create = Operator::creation(Operator::Spin::Down, 3);
  Operator annihilate = Operator::annihilation(Operator::Spin::Down, 3);
  FermionMonomial left(create);
  FermionMonomial right(annihilate);

  Expression result = anticommutator(left, right);

  Expression::container_type empty{};
  auto it_empty = result.terms().find(empty);
  CHECK(it_empty != result.terms().end());
  CHECK((it_empty->second) == (Expression::complex_type(1.0f, 0.0f)));
  CHECK((result.size()) == (1u));
}

TEST_CASE("commutator_expression_distributes_over_terms") {
  Operator a = Operator::creation(Operator::Spin::Up, 1);
  Operator b = Operator::creation(Operator::Spin::Up, 2);
  Operator annihilate = Operator::annihilation(Operator::Spin::Up, 1);
  FermionMonomial term_a(a);
  FermionMonomial term_b(b);
  FermionMonomial term_c(annihilate);

  Expression left({term_a, term_b});
  Expression right(term_c);

  Expression result = commutator(left, right);
  Expression expected = commutator(term_a, term_c);
  expected += commutator(term_b, term_c);

  CHECK((result.size()) == (expected.size()));
  for (const auto& [ops, coeff] : expected.terms()) {
    auto it = result.terms().find(ops);
    CHECK(it != result.terms().end());
    CHECK((it->second) == (coeff));
  }
}

TEST_CASE("bch_order_one_matches_first_commutator_term") {
  Operator create = Operator::creation(Operator::Spin::Down, 3);
  Operator annihilate = Operator::annihilation(Operator::Spin::Down, 3);
  Expression A{FermionMonomial(create)};
  Expression B{FermionMonomial(annihilate)};

  const Expression::complex_type::value_type lambda = 0.5;
  Expression result = BCH(A, B, lambda, 1);

  Expression expected = B + (commutator(A, B) * lambda);
  CHECK((result.size()) == (expected.size()));
  for (const auto& [ops, coeff] : expected.terms()) {
    auto it = result.terms().find(ops);
    CHECK(it != result.terms().end());
    CHECK((it->second) == (coeff));
  }
}

TEST_CASE("bch_zero_lambda_returns_b") {
  Operator create = Operator::creation(Operator::Spin::Up, 2);
  Operator annihilate = Operator::annihilation(Operator::Spin::Up, 2);
  Expression A{FermionMonomial(create)};
  Expression B{FermionMonomial(annihilate)};

  Expression result = BCH(A, B, 0.0, 5);

  CHECK((result.size()) == (B.size()));
  for (const auto& [ops, coeff] : B.terms()) {
    auto it = result.terms().find(ops);
    CHECK(it != result.terms().end());
    CHECK((it->second) == (coeff));
  }
}

TEST_CASE("bch_identity_generator_leaves_b") {
  Operator create = Operator::creation(Operator::Spin::Down, 1);
  Expression A(Expression::complex_type(1.0f, 0.0f));
  Expression B{FermionMonomial(create)};

  Expression result = BCH(A, B, 0.75, 4);

  CHECK((result.size()) == (B.size()));
  for (const auto& [ops, coeff] : B.terms()) {
    auto it = result.terms().find(ops);
    CHECK(it != result.terms().end());
    CHECK((it->second) == (coeff));
  }
}
