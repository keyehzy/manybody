#include <catch2/catch.hpp>

#include "algebra/fermion/term.h"

TEST_CASE("term_default_is_identity") {
  FermionMonomial term;
  CHECK((term.c) == (FermionMonomial::complex_type(1.0, 0.0)));
  CHECK((term.size()) == (0u));
}

TEST_CASE("term_construct_from_operator") {
  Operator op = Operator::creation(Operator::Spin::Up, 3);
  FermionMonomial term(op);
  CHECK((term.size()) == (1u));
  CHECK((*term.operators.begin()) == (op));
}

TEST_CASE("term_construct_from_complex") {
  FermionMonomial term(FermionMonomial::complex_type(2.0, -1.5));
  CHECK((term.c) == (FermionMonomial::complex_type(2.0, -1.5)));
  CHECK((term.size()) == (0u));
}

TEST_CASE("term_construct_from_initializer_list") {
  Operator a = Operator::creation(Operator::Spin::Up, 1);
  Operator b = Operator::annihilation(Operator::Spin::Down, 2);
  FermionMonomial term(FermionMonomial::complex_type(0.5, 0.5), {a, b});
  CHECK((term.c) == (FermionMonomial::complex_type(0.5, 0.5)));
  CHECK((term.size()) == (2u));
  CHECK((term.operators[0]) == (a));
  CHECK((term.operators[1]) == (b));
}

TEST_CASE("term_adjoint_conjugates_and_reverses") {
  Operator a = Operator::creation(Operator::Spin::Up, 4);
  Operator b = Operator::annihilation(Operator::Spin::Down, 5);
  FermionMonomial term(FermionMonomial::complex_type(1.0, 2.0), {a, b});
  FermionMonomial adj = adjoint(term);

  CHECK((adj.c) == (FermionMonomial::complex_type(1.0, -2.0)));
  CHECK((adj.size()) == (2u));
  CHECK((adj.operators[0]) == (b.adjoint()));
  CHECK((adj.operators[1]) == (a.adjoint()));
}

TEST_CASE("term_multiply_term_combines_coeff_and_ops") {
  Operator a = Operator::creation(Operator::Spin::Up, 1);
  Operator b = Operator::annihilation(Operator::Spin::Up, 2);
  Operator c = Operator::creation(Operator::Spin::Down, 3);
  FermionMonomial lhs(FermionMonomial::complex_type(2.0, 0.0), {a});
  FermionMonomial rhs(FermionMonomial::complex_type(0.0, 1.0), {b, c});

  lhs *= rhs;

  CHECK((lhs.c) == (FermionMonomial::complex_type(0.0, 2.0)));
  CHECK((lhs.size()) == (3u));
  CHECK((lhs.operators[0]) == (a));
  CHECK((lhs.operators[1]) == (b));
  CHECK((lhs.operators[2]) == (c));
}

TEST_CASE("term_multiply_operator_appends") {
  Operator a = Operator::creation(Operator::Spin::Up, 7);
  Operator b = Operator::annihilation(Operator::Spin::Down, 8);
  FermionMonomial term(a);
  term *= b;

  CHECK((term.size()) == (2u));
  CHECK((term.operators[0]) == (a));
  CHECK((term.operators[1]) == (b));
}

TEST_CASE("term_scale_and_divide") {
  FermionMonomial term;
  term *= FermionMonomial::complex_type(2.0, 0.0);
  term /= FermionMonomial::complex_type(4.0, 0.0);
  CHECK((term.c) == (FermionMonomial::complex_type(0.5, 0.0)));
}

TEST_CASE("term_binary_operator_term_term") {
  Operator a = Operator::creation(Operator::Spin::Up, 2);
  Operator b = Operator::annihilation(Operator::Spin::Down, 3);
  FermionMonomial left(FermionMonomial::complex_type(2.0, 0.0), {a});
  FermionMonomial right(FermionMonomial::complex_type(0.0, 1.0), {b});

  FermionMonomial result = left * right;

  CHECK((result.c) == (FermionMonomial::complex_type(0.0, 2.0)));
  CHECK((result.size()) == (2u));
  CHECK((result.operators[0]) == (a));
  CHECK((result.operators[1]) == (b));
}

TEST_CASE("term_binary_operator_term_operator") {
  Operator a = Operator::creation(Operator::Spin::Up, 5);
  Operator b = Operator::annihilation(Operator::Spin::Up, 6);
  FermionMonomial term(a);

  FermionMonomial result = term * b;

  CHECK((result.size()) == (2u));
  CHECK((result.operators[0]) == (a));
  CHECK((result.operators[1]) == (b));
}

TEST_CASE("term_binary_operator_operator_term") {
  Operator a = Operator::creation(Operator::Spin::Down, 4);
  Operator b = Operator::annihilation(Operator::Spin::Down, 1);
  FermionMonomial term(b);

  FermionMonomial result = a * term;

  CHECK((result.size()) == (2u));
  CHECK((result.operators[0]) == (a));
  CHECK((result.operators[1]) == (b));
}

TEST_CASE("term_binary_operator_term_complex") {
  Operator a = Operator::creation(Operator::Spin::Up, 9);
  FermionMonomial term(FermionMonomial::complex_type(3.0, 0.0), {a});

  FermionMonomial result = term * FermionMonomial::complex_type(0.0, 2.0);

  CHECK((result.c) == (FermionMonomial::complex_type(0.0, 6.0)));
  CHECK((result.size()) == (1u));
  CHECK((*result.operators.begin()) == (a));
}

TEST_CASE("term_binary_operator_complex_term") {
  Operator a = Operator::annihilation(Operator::Spin::Up, 8);
  FermionMonomial term(FermionMonomial::complex_type(0.0, 2.0), {a});

  FermionMonomial result = FermionMonomial::complex_type(0.5, 0.0) * term;

  CHECK((result.c) == (FermionMonomial::complex_type(0.0, 1.0)));
  CHECK((result.size()) == (1u));
  CHECK((*result.operators.begin()) == (a));
}

TEST_CASE("term_binary_operator_term_divide") {
  Operator a = Operator::creation(Operator::Spin::Down, 2);
  FermionMonomial term(FermionMonomial::complex_type(2.0, 0.0), {a});

  FermionMonomial result = term / FermionMonomial::complex_type(4.0, 0.0);

  CHECK((result.c) == (FermionMonomial::complex_type(0.5, 0.0)));
  CHECK((result.size()) == (1u));
  CHECK((*result.operators.begin()) == (a));
}

TEST_CASE("term_creation_helper") {
  FermionMonomial term = creation(Operator::Spin::Up, 3);

  CHECK((term.c) == (FermionMonomial::complex_type(1.0, 0.0)));
  CHECK((term.size()) == (1u));
  CHECK((*term.operators.begin()) == (Operator::creation(Operator::Spin::Up, 3)));
}

TEST_CASE("term_annihilation_helper") {
  FermionMonomial term = annihilation(Operator::Spin::Down, 5);

  CHECK((term.c) == (FermionMonomial::complex_type(1.0, 0.0)));
  CHECK((term.size()) == (1u));
  CHECK((*term.operators.begin()) == (Operator::annihilation(Operator::Spin::Down, 5)));
}

TEST_CASE("term_one_body_helper") {
  FermionMonomial term = one_body(Operator::Spin::Up, 1, Operator::Spin::Down, 2);

  CHECK((term.c) == (FermionMonomial::complex_type(1.0, 0.0)));
  CHECK((term.size()) == (2u));
  CHECK((term.operators[0]) == (Operator::creation(Operator::Spin::Up, 1)));
  CHECK((term.operators[1]) == (Operator::annihilation(Operator::Spin::Down, 2)));
}

TEST_CASE("term_two_body_helper") {
  FermionMonomial term = two_body(Operator::Spin::Up, 1, Operator::Spin::Down, 2,
                                  Operator::Spin::Up, 3, Operator::Spin::Down, 4);

  CHECK((term.c) == (FermionMonomial::complex_type(1.0, 0.0)));
  CHECK((term.size()) == (4u));
  CHECK((term.operators[0]) == (Operator::creation(Operator::Spin::Up, 1)));
  CHECK((term.operators[1]) == (Operator::creation(Operator::Spin::Down, 2)));
  CHECK((term.operators[2]) == (Operator::annihilation(Operator::Spin::Up, 3)));
  CHECK((term.operators[3]) == (Operator::annihilation(Operator::Spin::Down, 4)));
}

TEST_CASE("term_density_helper") {
  FermionMonomial term = density(Operator::Spin::Down, 7);

  CHECK((term.c) == (FermionMonomial::complex_type(1.0, 0.0)));
  CHECK((term.size()) == (2u));
  CHECK((term.operators[0]) == (Operator::creation(Operator::Spin::Down, 7)));
  CHECK((term.operators[1]) == (Operator::annihilation(Operator::Spin::Down, 7)));
}

TEST_CASE("term_density_density_helper") {
  FermionMonomial term = density_density(Operator::Spin::Up, 1, Operator::Spin::Down, 2);

  CHECK((term.c) == (FermionMonomial::complex_type(1.0, 0.0)));
  CHECK((term.size()) == (4u));
  CHECK((term.operators[0]) == (Operator::creation(Operator::Spin::Up, 1)));
  CHECK((term.operators[1]) == (Operator::annihilation(Operator::Spin::Up, 1)));
  CHECK((term.operators[2]) == (Operator::creation(Operator::Spin::Down, 2)));
  CHECK((term.operators[3]) == (Operator::annihilation(Operator::Spin::Down, 2)));
}

TEST_CASE("term_is_diagonal_empty_is_true") {
  FermionMonomial term;
  CHECK(is_diagonal(term));
}

TEST_CASE("term_is_diagonal_single_operator_is_false") {
  FermionMonomial term = creation(Operator::Spin::Up, 0);
  CHECK(!is_diagonal(term));
}

TEST_CASE("term_is_diagonal_matching_pair_is_true") {
  FermionMonomial term = density(Operator::Spin::Down, 3);
  CHECK(is_diagonal(term));
}

TEST_CASE("term_is_diagonal_mismatched_pair_is_false") {
  FermionMonomial term = one_body(Operator::Spin::Up, 1, Operator::Spin::Up, 2);
  CHECK(!is_diagonal(term));
}

TEST_CASE("term_is_diagonal_multiple_pairs_is_true") {
  FermionMonomial term = density_density(Operator::Spin::Up, 1, Operator::Spin::Down, 2);
  CHECK(is_diagonal(term));
}
