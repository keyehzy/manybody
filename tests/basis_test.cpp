#include "basis.h"

#include "framework.h"

TEST(basis_restrict_single_particle) {
  Basis basis(1, 1, Basis::Strategy::Restrict);

  EXPECT_EQ(basis.set.size(), 2u);
  EXPECT_TRUE(basis.set.contains({Operator::creation(Operator::Spin::Up, 0)}));
  EXPECT_TRUE(basis.set.contains({Operator::creation(Operator::Spin::Down, 0)}));
}

TEST(basis_restrict_two_particles_one_orbital) {
  Basis basis(1, 2, Basis::Strategy::Restrict);

  EXPECT_EQ(basis.set.size(), 1u);
  EXPECT_TRUE(basis.set.contains(
      {Operator::creation(Operator::Spin::Up, 0), Operator::creation(Operator::Spin::Down, 0)}));
}

TEST(basis_all_orders_by_size) {
  Basis basis(1, 2, Basis::Strategy::All);

  EXPECT_EQ(basis.set.size(), 3u);
  EXPECT_EQ(basis.set[0].size(), 1u);
  EXPECT_EQ(basis.set[1].size(), 1u);
  EXPECT_EQ(basis.set[2].size(), 2u);
  EXPECT_TRUE(basis.set.contains({Operator::creation(Operator::Spin::Up, 0)}));
  EXPECT_TRUE(basis.set.contains({Operator::creation(Operator::Spin::Down, 0)}));
  EXPECT_TRUE(basis.set.contains(
      {Operator::creation(Operator::Spin::Up, 0), Operator::creation(Operator::Spin::Down, 0)}));
}
