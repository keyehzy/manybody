#include "operator.h"

#include "framework.h"

TEST(operator_bits_roundtrip) {
  Operator op(Operator::Type::Creation, Operator::Spin::Down, 12);
  EXPECT_EQ(op.type(), Operator::Type::Creation);
  EXPECT_EQ(op.spin(), Operator::Spin::Down);
  EXPECT_EQ(op.value(), 12u);
}

TEST(operator_bits_layout) {
  Operator op(Operator::Type::Annihilation, Operator::Spin::Down, 45);
  const Operator::ubyte expected = static_cast<Operator::ubyte>(
      Operator::kTypeBit | Operator::kSpinBit | (45u & Operator::kValueMask));
  EXPECT_EQ(op.data, expected);
  EXPECT_EQ(op.value(), 45u);
}

TEST(operator_adjoint) {
  Operator a(Operator::Type::Creation, Operator::Spin::Up, 3);
  Operator b = a.adjoint();
  EXPECT_EQ(b.type(), Operator::Type::Annihilation);
  EXPECT_EQ(b.spin(), Operator::Spin::Up);
  EXPECT_EQ(b.value(), 3u);
}

TEST(operator_adjoint_involution) {
  Operator a(Operator::Type::Annihilation, Operator::Spin::Down, 7);
  Operator b = a.adjoint().adjoint();
  EXPECT_EQ(b, a);
}

TEST(operator_commutes) {
  Operator create_up(Operator::Type::Creation, Operator::Spin::Up, 1);
  Operator annihilate_up(Operator::Type::Annihilation, Operator::Spin::Up, 1);
  Operator create_down(Operator::Type::Creation, Operator::Spin::Down, 1);

  EXPECT_TRUE(!create_up.commutes(annihilate_up));
  EXPECT_TRUE(create_up.commutes(create_down));
}

TEST(operator_commutes_for_different_value_or_spin) {
  Operator create_up(Operator::Type::Creation, Operator::Spin::Up, 1);
  Operator create_up_other_value(Operator::Type::Creation, Operator::Spin::Up,
                                 2);
  Operator annihilate_down(Operator::Type::Annihilation, Operator::Spin::Down,
                           1);

  EXPECT_TRUE(create_up.commutes(create_up));
  EXPECT_TRUE(create_up.commutes(create_up_other_value));
  EXPECT_TRUE(create_up.commutes(annihilate_down));
}

TEST(operator_ordering_uses_packed_bits) {
  Operator create_up_1(Operator::Type::Creation, Operator::Spin::Up, 1);
  Operator create_up_2(Operator::Type::Creation, Operator::Spin::Up, 2);
  Operator annihilate_up_1(Operator::Type::Annihilation, Operator::Spin::Up, 1);

  EXPECT_TRUE(create_up_1 < create_up_2);
  EXPECT_TRUE(create_up_1 < annihilate_up_1);
}
