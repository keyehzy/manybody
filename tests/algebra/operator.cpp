#include "algebra/operator.h"

#include <unordered_set>

#include "framework.h"

TEST(operator_bits_roundtrip) {
  Operator op = Operator::creation(Operator::Spin::Down, 12);
  EXPECT_EQ(op.type(), Operator::Type::Creation);
  EXPECT_EQ(op.spin(), Operator::Spin::Down);
  EXPECT_EQ(op.value(), 12u);
}

TEST(operator_bits_layout) {
  Operator op = Operator::annihilation(Operator::Spin::Down, 45);
  const Operator::ubyte expected = static_cast<Operator::ubyte>(
      Operator::kTypeBit | Operator::kSpinBit | (45u & Operator::kValueMask));
  EXPECT_EQ(op.data, expected);
  EXPECT_EQ(op.value(), 45u);
}

TEST(operator_creation_sets_type) {
  Operator op = Operator::creation(Operator::Spin::Up, 9);
  EXPECT_EQ(op.type(), Operator::Type::Creation);
  EXPECT_EQ(op.spin(), Operator::Spin::Up);
  EXPECT_EQ(op.value(), 9u);
}

TEST(operator_annihilation_sets_type) {
  Operator op = Operator::annihilation(Operator::Spin::Down, 5);
  EXPECT_EQ(op.type(), Operator::Type::Annihilation);
  EXPECT_EQ(op.spin(), Operator::Spin::Down);
  EXPECT_EQ(op.value(), 5u);
}

TEST(operator_adjoint) {
  Operator a = Operator::creation(Operator::Spin::Up, 3);
  Operator b = a.adjoint();
  EXPECT_EQ(b.type(), Operator::Type::Annihilation);
  EXPECT_EQ(b.spin(), Operator::Spin::Up);
  EXPECT_EQ(b.value(), 3u);
}

TEST(operator_adjoint_involution) {
  Operator a = Operator::annihilation(Operator::Spin::Down, 7);
  Operator b = a.adjoint().adjoint();
  EXPECT_EQ(b, a);
}

TEST(operator_flip) {
  Operator a = Operator::creation(Operator::Spin::Up, 4);
  Operator b = a.flip();
  EXPECT_EQ(b.type(), Operator::Type::Creation);
  EXPECT_EQ(b.spin(), Operator::Spin::Down);
  EXPECT_EQ(b.value(), 4u);
}

TEST(operator_flip_involution) {
  Operator a = Operator::annihilation(Operator::Spin::Down, 13);
  Operator b = a.flip().flip();
  EXPECT_EQ(b, a);
}

TEST(operator_commutes) {
  Operator create_up = Operator::creation(Operator::Spin::Up, 1);
  Operator annihilate_up = Operator::annihilation(Operator::Spin::Up, 1);
  Operator create_down = Operator::creation(Operator::Spin::Down, 1);

  EXPECT_TRUE(!create_up.commutes(annihilate_up));
  EXPECT_TRUE(create_up.commutes(create_down));
}

TEST(operator_commutes_for_different_value_or_spin) {
  Operator create_up = Operator::creation(Operator::Spin::Up, 1);
  Operator create_up_other_value = Operator::creation(Operator::Spin::Up, 2);
  Operator annihilate_down = Operator::annihilation(Operator::Spin::Down, 1);

  EXPECT_TRUE(create_up.commutes(create_up));
  EXPECT_TRUE(create_up.commutes(create_up_other_value));
  EXPECT_TRUE(create_up.commutes(annihilate_down));
}

TEST(operator_ordering_uses_packed_bits) {
  Operator create_up_1 = Operator::creation(Operator::Spin::Up, 1);
  Operator create_up_2 = Operator::creation(Operator::Spin::Up, 2);
  Operator annihilate_up_1 = Operator::annihilation(Operator::Spin::Up, 1);

  EXPECT_TRUE(create_up_1 < create_up_2);
  EXPECT_TRUE(create_up_1 < annihilate_up_1);
}

TEST(operator_hash_matches_data) {
  Operator op = Operator::annihilation(Operator::Spin::Down, 31);
  std::hash<Operator> hasher;
  EXPECT_EQ(hasher(op), static_cast<size_t>(op.data));
}

TEST(operator_hash_works_in_unordered_set) {
  std::unordered_set<Operator> ops;
  Operator a = Operator::creation(Operator::Spin::Up, 2);
  Operator b = Operator::annihilation(Operator::Spin::Down, 2);

  ops.insert(a);
  ops.insert(b);

  EXPECT_TRUE(ops.find(a) != ops.end());
  EXPECT_TRUE(ops.find(b) != ops.end());
}
