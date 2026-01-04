#include "commutator.h"

#include "framework.h"

TEST(commutator_commuting_creations_doubles_ordered_term) {
  Operator a = Operator::creation(Operator::Spin::Up, 1);
  Operator b = Operator::creation(Operator::Spin::Up, 2);
  Term left(a);
  Term right(b);

  Expression result = commutator(left, right);

  Expression::container_type ordered{a, b};
  auto it = result.hashmap.find(ordered);
  EXPECT_TRUE(it != result.hashmap.end());
  EXPECT_EQ(it->second, Expression::complex_type(2.0f, 0.0f));
  EXPECT_EQ(result.size(), 1u);
}

TEST(anticommutator_commuting_creations_vanishes) {
  Operator a = Operator::creation(Operator::Spin::Up, 1);
  Operator b = Operator::creation(Operator::Spin::Up, 2);
  Term left(a);
  Term right(b);

  Expression result = anticommutator(left, right);
  EXPECT_EQ(result.size(), 0u);
}

TEST(commutator_creation_annihilation_same_orbital) {
  Operator create = Operator::creation(Operator::Spin::Down, 3);
  Operator annihilate = Operator::annihilation(Operator::Spin::Down, 3);
  Term left(create);
  Term right(annihilate);

  Expression result = commutator(left, right);

  Expression::container_type empty{};
  auto it_empty = result.hashmap.find(empty);
  EXPECT_TRUE(it_empty != result.hashmap.end());
  EXPECT_EQ(it_empty->second, Expression::complex_type(-1.0f, 0.0f));

  Expression::container_type ordered{create, annihilate};
  auto it_ordered = result.hashmap.find(ordered);
  EXPECT_TRUE(it_ordered != result.hashmap.end());
  EXPECT_EQ(it_ordered->second, Expression::complex_type(2.0f, 0.0f));
  EXPECT_EQ(result.size(), 2u);
}

TEST(anticommutator_creation_annihilation_same_orbital_is_identity) {
  Operator create = Operator::creation(Operator::Spin::Down, 3);
  Operator annihilate = Operator::annihilation(Operator::Spin::Down, 3);
  Term left(create);
  Term right(annihilate);

  Expression result = anticommutator(left, right);

  Expression::container_type empty{};
  auto it_empty = result.hashmap.find(empty);
  EXPECT_TRUE(it_empty != result.hashmap.end());
  EXPECT_EQ(it_empty->second, Expression::complex_type(1.0f, 0.0f));
  EXPECT_EQ(result.size(), 1u);
}

TEST(commutator_expression_distributes_over_terms) {
  Operator a = Operator::creation(Operator::Spin::Up, 1);
  Operator b = Operator::creation(Operator::Spin::Up, 2);
  Operator annihilate = Operator::annihilation(Operator::Spin::Up, 1);
  Term term_a(a);
  Term term_b(b);
  Term term_c(annihilate);

  Expression left({term_a, term_b});
  Expression right(term_c);

  Expression result = commutator(left, right);
  Expression expected = commutator(term_a, term_c);
  expected += commutator(term_b, term_c);

  EXPECT_EQ(result.size(), expected.size());
  for (const auto& [ops, coeff] : expected.hashmap) {
    auto it = result.hashmap.find(ops);
    EXPECT_TRUE(it != result.hashmap.end());
    EXPECT_EQ(it->second, coeff);
  }
}
