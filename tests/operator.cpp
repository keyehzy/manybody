#include "operator.h"

#include "framework.h"

TEST(operator_bits_roundtrip) {
  Operator op(Operator::Type::Creation, Operator::Spin::Down, 12);
  EXPECT_EQ(op.type(), Operator::Type::Creation);
  EXPECT_EQ(op.spin(), Operator::Spin::Down);
  EXPECT_EQ(op.value(), 12u);
}

TEST(operator_adjoint) {
  Operator a(Operator::Type::Creation, Operator::Spin::Up, 3);
  Operator b = a.adjoint();
  EXPECT_EQ(b.type(), Operator::Type::Annihilation);
  EXPECT_EQ(b.spin(), Operator::Spin::Up);
  EXPECT_EQ(b.value(), 3u);
}

TEST(operator_commutes) {
  Operator create_up(Operator::Type::Creation, Operator::Spin::Up, 1);
  Operator annihilate_up(Operator::Type::Annihilation, Operator::Spin::Up, 1);
  Operator create_down(Operator::Type::Creation, Operator::Spin::Down, 1);

  EXPECT_TRUE(!create_up.commutes(annihilate_up));
  EXPECT_TRUE(create_up.commutes(create_down));
}
