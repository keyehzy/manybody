#include "algebra/normal_order.h"
#include "framework.h"

TEST(normal_order_zero_coefficient_returns_empty) {
  NormalOrderer orderer;
  Term::container_type ops{};
  Expression result = orderer.normal_order(Term::complex_type(0.0f, 0.0f), ops);
  EXPECT_EQ(result.size(), 0u);
}

TEST(normal_order_single_operator_is_identity) {
  NormalOrderer orderer;
  Operator op = Operator::creation(Operator::Spin::Up, 1);
  Term term(op);
  Expression result = orderer.normal_order(term);

  Expression::container_type ops{op};
  auto it = result.hashmap.find(ops);
  EXPECT_TRUE(it != result.hashmap.end());
  EXPECT_EQ(it->second, Expression::complex_type(1.0f, 0.0f));
}

TEST(normal_order_commuting_swap_introduces_phase) {
  NormalOrderer orderer;
  Operator a = Operator::creation(Operator::Spin::Up, 1);
  Operator b = Operator::creation(Operator::Spin::Up, 2);
  Term term(Term::complex_type(1.0f, 0.0f), {b, a});
  Expression result = orderer.normal_order(term);

  Expression::container_type ordered{a, b};
  auto it = result.hashmap.find(ordered);
  EXPECT_TRUE(it != result.hashmap.end());
  EXPECT_EQ(it->second, Expression::complex_type(-1.0f, 0.0f));
  EXPECT_EQ(result.size(), 1u);
}

TEST(normal_order_non_commuting_pair_contracts) {
  NormalOrderer orderer;
  Operator create = Operator::creation(Operator::Spin::Down, 3);
  Operator annihilate = Operator::annihilation(Operator::Spin::Down, 3);
  Term term(Term::complex_type(1.0f, 0.0f), {annihilate, create});
  Expression result = orderer.normal_order(term);

  Expression::container_type empty{};
  auto it_empty = result.hashmap.find(empty);
  EXPECT_TRUE(it_empty != result.hashmap.end());
  EXPECT_EQ(it_empty->second, Expression::complex_type(1.0f, 0.0f));

  Expression::container_type ordered{create, annihilate};
  auto it_ordered = result.hashmap.find(ordered);
  EXPECT_TRUE(it_ordered != result.hashmap.end());
  EXPECT_EQ(it_ordered->second, Expression::complex_type(-1.0f, 0.0f));
  EXPECT_EQ(result.size(), 2u);
}

TEST(normal_order_consecutive_duplicates_vanish) {
  NormalOrderer orderer;
  Operator a = Operator::creation(Operator::Spin::Up, 4);
  Term term(Term::complex_type(1.0f, 0.0f), {a, a});
  Expression result = orderer.normal_order(term);
  EXPECT_EQ(result.size(), 0u);
}
