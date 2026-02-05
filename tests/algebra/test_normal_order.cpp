#include <catch2/catch.hpp>

#include "algebra/expression.h"

TEST_CASE("canonicalize_zero_coefficient_returns_empty") {
  FermionMonomial::container_type ops{};
  Expression result = canonicalize(FermionMonomial::complex_type(0.0f, 0.0f), ops);
  CHECK((result.size()) == (0u));
}

TEST_CASE("canonicalize_single_operator_is_identity") {
  Operator op = Operator::creation(Operator::Spin::Up, 1);
  FermionMonomial term(op);
  Expression result = canonicalize(term);

  Expression::container_type ops{op};
  auto it = result.terms().find(ops);
  CHECK(it != result.terms().end());
  CHECK((it->second) == (Expression::complex_type(1.0f, 0.0f)));
}

TEST_CASE("canonicalize_commuting_swap_introduces_phase") {
  Operator a = Operator::creation(Operator::Spin::Up, 1);
  Operator b = Operator::creation(Operator::Spin::Up, 2);
  FermionMonomial term(FermionMonomial::complex_type(1.0f, 0.0f), {b, a});
  Expression result = canonicalize(term);

  Expression::container_type ordered{a, b};
  auto it = result.terms().find(ordered);
  CHECK(it != result.terms().end());
  CHECK((it->second) == (Expression::complex_type(-1.0f, 0.0f)));
  CHECK((result.size()) == (1u));
}

TEST_CASE("canonicalize_non_commuting_pair_contracts") {
  Operator create = Operator::creation(Operator::Spin::Down, 3);
  Operator annihilate = Operator::annihilation(Operator::Spin::Down, 3);
  FermionMonomial term(FermionMonomial::complex_type(1.0f, 0.0f), {annihilate, create});
  Expression result = canonicalize(term);

  Expression::container_type empty{};
  auto it_empty = result.terms().find(empty);
  CHECK(it_empty != result.terms().end());
  CHECK((it_empty->second) == (Expression::complex_type(1.0f, 0.0f)));

  Expression::container_type ordered{create, annihilate};
  auto it_ordered = result.terms().find(ordered);
  CHECK(it_ordered != result.terms().end());
  CHECK((it_ordered->second) == (Expression::complex_type(-1.0f, 0.0f)));
  CHECK((result.size()) == (2u));
}

TEST_CASE("canonicalize_consecutive_duplicates_vanish") {
  Operator a = Operator::creation(Operator::Spin::Up, 4);
  FermionMonomial term(FermionMonomial::complex_type(1.0f, 0.0f), {a, a});
  Expression result = canonicalize(term);
  CHECK((result.size()) == (0u));
}
