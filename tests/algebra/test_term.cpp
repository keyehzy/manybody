#include <catch2/catch.hpp>

#include "algebra/boson/term.h"
#include "algebra/fermion/term.h"

TEST_CASE("term_default_is_identity") {
  FermionMonomial term;
  CHECK((term.c) == (FermionMonomial::complex_type(1.0, 0.0)));
  CHECK((term.size()) == (0u));
}

TEST_CASE("term_construct_from_operator") {
  FermionOperator op = FermionOperator::creation(FermionOperator::Spin::Up, 3);
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
  FermionOperator a = FermionOperator::creation(FermionOperator::Spin::Up, 1);
  FermionOperator b = FermionOperator::annihilation(FermionOperator::Spin::Down, 2);
  FermionMonomial term(FermionMonomial::complex_type(0.5, 0.5), {a, b});
  CHECK((term.c) == (FermionMonomial::complex_type(0.5, 0.5)));
  CHECK((term.size()) == (2u));
  CHECK((term.operators[0]) == (a));
  CHECK((term.operators[1]) == (b));
}

TEST_CASE("term_adjoint_conjugates_and_reverses") {
  FermionOperator a = FermionOperator::creation(FermionOperator::Spin::Up, 4);
  FermionOperator b = FermionOperator::annihilation(FermionOperator::Spin::Down, 5);
  FermionMonomial term(FermionMonomial::complex_type(1.0, 2.0), {a, b});
  FermionMonomial adj = adjoint(term);

  CHECK((adj.c) == (FermionMonomial::complex_type(1.0, -2.0)));
  CHECK((adj.size()) == (2u));
  CHECK((adj.operators[0]) == (b.adjoint()));
  CHECK((adj.operators[1]) == (a.adjoint()));
}

TEST_CASE("term_multiply_term_combines_coeff_and_ops") {
  FermionOperator a = FermionOperator::creation(FermionOperator::Spin::Up, 1);
  FermionOperator b = FermionOperator::annihilation(FermionOperator::Spin::Up, 2);
  FermionOperator c = FermionOperator::creation(FermionOperator::Spin::Down, 3);
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
  FermionOperator a = FermionOperator::creation(FermionOperator::Spin::Up, 7);
  FermionOperator b = FermionOperator::annihilation(FermionOperator::Spin::Down, 8);
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
  FermionOperator a = FermionOperator::creation(FermionOperator::Spin::Up, 2);
  FermionOperator b = FermionOperator::annihilation(FermionOperator::Spin::Down, 3);
  FermionMonomial left(FermionMonomial::complex_type(2.0, 0.0), {a});
  FermionMonomial right(FermionMonomial::complex_type(0.0, 1.0), {b});

  FermionMonomial result = left * right;

  CHECK((result.c) == (FermionMonomial::complex_type(0.0, 2.0)));
  CHECK((result.size()) == (2u));
  CHECK((result.operators[0]) == (a));
  CHECK((result.operators[1]) == (b));
}

TEST_CASE("term_binary_operator_term_operator") {
  FermionOperator a = FermionOperator::creation(FermionOperator::Spin::Up, 5);
  FermionOperator b = FermionOperator::annihilation(FermionOperator::Spin::Up, 6);
  FermionMonomial term(a);

  FermionMonomial result = term * b;

  CHECK((result.size()) == (2u));
  CHECK((result.operators[0]) == (a));
  CHECK((result.operators[1]) == (b));
}

TEST_CASE("term_binary_operator_operator_term") {
  FermionOperator a = FermionOperator::creation(FermionOperator::Spin::Down, 4);
  FermionOperator b = FermionOperator::annihilation(FermionOperator::Spin::Down, 1);
  FermionMonomial term(b);

  FermionMonomial result = a * term;

  CHECK((result.size()) == (2u));
  CHECK((result.operators[0]) == (a));
  CHECK((result.operators[1]) == (b));
}

TEST_CASE("term_binary_operator_term_complex") {
  FermionOperator a = FermionOperator::creation(FermionOperator::Spin::Up, 9);
  FermionMonomial term(FermionMonomial::complex_type(3.0, 0.0), {a});

  FermionMonomial result = term * FermionMonomial::complex_type(0.0, 2.0);

  CHECK((result.c) == (FermionMonomial::complex_type(0.0, 6.0)));
  CHECK((result.size()) == (1u));
  CHECK((*result.operators.begin()) == (a));
}

TEST_CASE("term_binary_operator_complex_term") {
  FermionOperator a = FermionOperator::annihilation(FermionOperator::Spin::Up, 8);
  FermionMonomial term(FermionMonomial::complex_type(0.0, 2.0), {a});

  FermionMonomial result = FermionMonomial::complex_type(0.5, 0.0) * term;

  CHECK((result.c) == (FermionMonomial::complex_type(0.0, 1.0)));
  CHECK((result.size()) == (1u));
  CHECK((*result.operators.begin()) == (a));
}

TEST_CASE("term_binary_operator_term_divide") {
  FermionOperator a = FermionOperator::creation(FermionOperator::Spin::Down, 2);
  FermionMonomial term(FermionMonomial::complex_type(2.0, 0.0), {a});

  FermionMonomial result = term / FermionMonomial::complex_type(4.0, 0.0);

  CHECK((result.c) == (FermionMonomial::complex_type(0.5, 0.0)));
  CHECK((result.size()) == (1u));
  CHECK((*result.operators.begin()) == (a));
}

TEST_CASE("term_creation_helper") {
  FermionMonomial term = creation(FermionOperator::Spin::Up, 3);

  CHECK((term.c) == (FermionMonomial::complex_type(1.0, 0.0)));
  CHECK((term.size()) == (1u));
  CHECK((*term.operators.begin()) == (FermionOperator::creation(FermionOperator::Spin::Up, 3)));
}

TEST_CASE("term_annihilation_helper") {
  FermionMonomial term = annihilation(FermionOperator::Spin::Down, 5);

  CHECK((term.c) == (FermionMonomial::complex_type(1.0, 0.0)));
  CHECK((term.size()) == (1u));
  CHECK((*term.operators.begin()) ==
        (FermionOperator::annihilation(FermionOperator::Spin::Down, 5)));
}

TEST_CASE("term_one_body_helper") {
  FermionMonomial term = one_body(FermionOperator::Spin::Up, 1, FermionOperator::Spin::Down, 2);

  CHECK((term.c) == (FermionMonomial::complex_type(1.0, 0.0)));
  CHECK((term.size()) == (2u));
  CHECK((term.operators[0]) == (FermionOperator::creation(FermionOperator::Spin::Up, 1)));
  CHECK((term.operators[1]) == (FermionOperator::annihilation(FermionOperator::Spin::Down, 2)));
}

TEST_CASE("term_two_body_helper") {
  FermionMonomial term = two_body(FermionOperator::Spin::Up, 1, FermionOperator::Spin::Down, 2,
                                  FermionOperator::Spin::Up, 3, FermionOperator::Spin::Down, 4);

  CHECK((term.c) == (FermionMonomial::complex_type(1.0, 0.0)));
  CHECK((term.size()) == (4u));
  CHECK((term.operators[0]) == (FermionOperator::creation(FermionOperator::Spin::Up, 1)));
  CHECK((term.operators[1]) == (FermionOperator::creation(FermionOperator::Spin::Down, 2)));
  CHECK((term.operators[2]) == (FermionOperator::annihilation(FermionOperator::Spin::Up, 3)));
  CHECK((term.operators[3]) == (FermionOperator::annihilation(FermionOperator::Spin::Down, 4)));
}

TEST_CASE("term_density_helper") {
  FermionMonomial term = density(FermionOperator::Spin::Down, 7);

  CHECK((term.c) == (FermionMonomial::complex_type(1.0, 0.0)));
  CHECK((term.size()) == (2u));
  CHECK((term.operators[0]) == (FermionOperator::creation(FermionOperator::Spin::Down, 7)));
  CHECK((term.operators[1]) == (FermionOperator::annihilation(FermionOperator::Spin::Down, 7)));
}

TEST_CASE("term_density_density_helper") {
  FermionMonomial term =
      density_density(FermionOperator::Spin::Up, 1, FermionOperator::Spin::Down, 2);

  CHECK((term.c) == (FermionMonomial::complex_type(1.0, 0.0)));
  CHECK((term.size()) == (4u));
  CHECK((term.operators[0]) == (FermionOperator::creation(FermionOperator::Spin::Up, 1)));
  CHECK((term.operators[1]) == (FermionOperator::annihilation(FermionOperator::Spin::Up, 1)));
  CHECK((term.operators[2]) == (FermionOperator::creation(FermionOperator::Spin::Down, 2)));
  CHECK((term.operators[3]) == (FermionOperator::annihilation(FermionOperator::Spin::Down, 2)));
}

TEST_CASE("term_is_diagonal_empty_is_true") {
  FermionMonomial term;
  CHECK(is_diagonal(term));
}

TEST_CASE("term_is_diagonal_single_operator_is_false") {
  FermionMonomial term = creation(FermionOperator::Spin::Up, 0);
  CHECK(!is_diagonal(term));
}

TEST_CASE("term_is_diagonal_matching_pair_is_true") {
  FermionMonomial term = density(FermionOperator::Spin::Down, 3);
  CHECK(is_diagonal(term));
}

TEST_CASE("term_is_diagonal_mismatched_pair_is_false") {
  FermionMonomial term = one_body(FermionOperator::Spin::Up, 1, FermionOperator::Spin::Up, 2);
  CHECK(!is_diagonal(term));
}

TEST_CASE("term_is_diagonal_multiple_pairs_is_true") {
  FermionMonomial term =
      density_density(FermionOperator::Spin::Up, 1, FermionOperator::Spin::Down, 2);
  CHECK(is_diagonal(term));
}

TEST_CASE("boson_term_single_operator_helpers") {
  BosonMonomial create = boson::creation(BosonOperator::Spin::Up, 3);
  BosonMonomial destroy = boson::annihilation(BosonOperator::Spin::Down, 5);

  CHECK((create.c) == (BosonMonomial::complex_type(1.0, 0.0)));
  CHECK((create.size()) == (1u));
  CHECK((*create.operators.begin()) == (BosonOperator::creation(BosonOperator::Spin::Up, 3)));

  CHECK((destroy.c) == (BosonMonomial::complex_type(1.0, 0.0)));
  CHECK((destroy.size()) == (1u));
  CHECK((*destroy.operators.begin()) ==
        (BosonOperator::annihilation(BosonOperator::Spin::Down, 5)));
}

TEST_CASE("boson_term_many_body_helpers") {
  BosonMonomial one_body_term =
      boson::one_body(BosonOperator::Spin::Up, 1, BosonOperator::Spin::Down, 2);
  BosonMonomial two_body_term =
      boson::two_body(BosonOperator::Spin::Up, 1, BosonOperator::Spin::Down, 2,
                      BosonOperator::Spin::Up, 3, BosonOperator::Spin::Down, 4);

  CHECK((one_body_term.size()) == (2u));
  CHECK((one_body_term.operators[0]) == (BosonOperator::creation(BosonOperator::Spin::Up, 1)));
  CHECK((one_body_term.operators[1]) ==
        (BosonOperator::annihilation(BosonOperator::Spin::Down, 2)));

  CHECK((two_body_term.size()) == (4u));
  CHECK((two_body_term.operators[0]) == (BosonOperator::creation(BosonOperator::Spin::Up, 1)));
  CHECK((two_body_term.operators[1]) == (BosonOperator::creation(BosonOperator::Spin::Down, 2)));
  CHECK((two_body_term.operators[2]) == (BosonOperator::annihilation(BosonOperator::Spin::Up, 3)));
  CHECK((two_body_term.operators[3]) ==
        (BosonOperator::annihilation(BosonOperator::Spin::Down, 4)));
}

TEST_CASE("boson_term_number_operator_helper") {
  BosonMonomial number = boson::number_op(BosonOperator::Spin::Down, 7);

  CHECK((number.c) == (BosonMonomial::complex_type(1.0, 0.0)));
  CHECK((number.size()) == (2u));
  CHECK((number.operators[0]) == (BosonOperator::creation(BosonOperator::Spin::Down, 7)));
  CHECK((number.operators[1]) == (BosonOperator::annihilation(BosonOperator::Spin::Down, 7)));
  CHECK(is_diagonal(number));
}
