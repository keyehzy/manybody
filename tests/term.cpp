#include "term.h"

#include "framework.h"

TEST(term_default_is_identity) {
  Term term;
  EXPECT_EQ(term.c, Term::complex_type(1.0f, 0.0f));
  EXPECT_EQ(term.size(), 0u);
}

TEST(term_construct_from_operator) {
  Operator op = Operator::creation(Operator::Spin::Up, 3);
  Term term(op);
  EXPECT_EQ(term.size(), 1u);
  EXPECT_EQ(*term.operators.begin(), op);
}

TEST(term_construct_from_complex) {
  Term term(Term::complex_type(2.0f, -1.5f));
  EXPECT_EQ(term.c, Term::complex_type(2.0f, -1.5f));
  EXPECT_EQ(term.size(), 0u);
}

TEST(term_construct_from_initializer_list) {
  Operator a = Operator::creation(Operator::Spin::Up, 1);
  Operator b = Operator::annihilation(Operator::Spin::Down, 2);
  Term term(Term::complex_type(0.5f, 0.25f), {a, b});
  EXPECT_EQ(term.c, Term::complex_type(0.5f, 0.25f));
  EXPECT_EQ(term.size(), 2u);
  auto it = term.operators.begin();
  EXPECT_EQ(*it++, a);
  EXPECT_EQ(*it++, b);
}

TEST(term_adjoint_conjugates_and_reverses) {
  Operator a = Operator::creation(Operator::Spin::Up, 4);
  Operator b = Operator::annihilation(Operator::Spin::Down, 5);
  Term term(Term::complex_type(1.0f, 2.0f), {a, b});
  Term adj = term.adjoint();

  EXPECT_EQ(adj.c, Term::complex_type(1.0f, -2.0f));
  EXPECT_EQ(adj.size(), 2u);
  auto it = adj.operators.begin();
  EXPECT_EQ(*it++, b.adjoint());
  EXPECT_EQ(*it++, a.adjoint());
}

TEST(term_multiply_term_combines_coeff_and_ops) {
  Operator a = Operator::creation(Operator::Spin::Up, 1);
  Operator b = Operator::annihilation(Operator::Spin::Up, 2);
  Operator c = Operator::creation(Operator::Spin::Down, 3);
  Term lhs(Term::complex_type(2.0f, 0.0f), {a});
  Term rhs(Term::complex_type(0.0f, 1.0f), {b, c});

  lhs *= rhs;

  EXPECT_EQ(lhs.c, Term::complex_type(0.0f, 2.0f));
  EXPECT_EQ(lhs.size(), 3u);
  auto it = lhs.operators.begin();
  EXPECT_EQ(*it++, a);
  EXPECT_EQ(*it++, b);
  EXPECT_EQ(*it++, c);
}

TEST(term_multiply_operator_appends) {
  Operator a = Operator::creation(Operator::Spin::Up, 7);
  Operator b = Operator::annihilation(Operator::Spin::Down, 8);
  Term term(a);
  term *= b;

  EXPECT_EQ(term.size(), 2u);
  auto it = term.operators.begin();
  EXPECT_EQ(*it++, a);
  EXPECT_EQ(*it++, b);
}

TEST(term_scale_and_divide) {
  Term term;
  term *= Term::complex_type(2.0f, 0.0f);
  term /= Term::complex_type(4.0f, 0.0f);
  EXPECT_EQ(term.c, Term::complex_type(0.5f, 0.0f));
}

TEST(term_binary_operator_term_term) {
  Operator a = Operator::creation(Operator::Spin::Up, 2);
  Operator b = Operator::annihilation(Operator::Spin::Down, 3);
  Term left(Term::complex_type(2.0f, 0.0f), {a});
  Term right(Term::complex_type(0.0f, 1.0f), {b});

  Term result = left * right;

  EXPECT_EQ(result.c, Term::complex_type(0.0f, 2.0f));
  EXPECT_EQ(result.size(), 2u);
  auto it = result.operators.begin();
  EXPECT_EQ(*it++, a);
  EXPECT_EQ(*it++, b);
}

TEST(term_binary_operator_term_operator) {
  Operator a = Operator::creation(Operator::Spin::Up, 5);
  Operator b = Operator::annihilation(Operator::Spin::Up, 6);
  Term term(a);

  Term result = term * b;

  EXPECT_EQ(result.size(), 2u);
  auto it = result.operators.begin();
  EXPECT_EQ(*it++, a);
  EXPECT_EQ(*it++, b);
}

TEST(term_binary_operator_operator_term) {
  Operator a = Operator::creation(Operator::Spin::Down, 4);
  Operator b = Operator::annihilation(Operator::Spin::Down, 1);
  Term term(b);

  Term result = a * term;

  EXPECT_EQ(result.size(), 2u);
  auto it = result.operators.begin();
  EXPECT_EQ(*it++, a);
  EXPECT_EQ(*it++, b);
}

TEST(term_binary_operator_term_complex) {
  Operator a = Operator::creation(Operator::Spin::Up, 9);
  Term term(Term::complex_type(3.0f, 0.0f), {a});

  Term result = term * Term::complex_type(0.0f, 2.0f);

  EXPECT_EQ(result.c, Term::complex_type(0.0f, 6.0f));
  EXPECT_EQ(result.size(), 1u);
  EXPECT_EQ(*result.operators.begin(), a);
}

TEST(term_binary_operator_complex_term) {
  Operator a = Operator::annihilation(Operator::Spin::Up, 8);
  Term term(Term::complex_type(0.0f, 2.0f), {a});

  Term result = Term::complex_type(0.5f, 0.0f) * term;

  EXPECT_EQ(result.c, Term::complex_type(0.0f, 1.0f));
  EXPECT_EQ(result.size(), 1u);
  EXPECT_EQ(*result.operators.begin(), a);
}

TEST(term_binary_operator_term_divide) {
  Operator a = Operator::creation(Operator::Spin::Down, 2);
  Term term(Term::complex_type(2.0f, 0.0f), {a});

  Term result = term / Term::complex_type(4.0f, 0.0f);

  EXPECT_EQ(result.c, Term::complex_type(0.5f, 0.0f));
  EXPECT_EQ(result.size(), 1u);
  EXPECT_EQ(*result.operators.begin(), a);
}
