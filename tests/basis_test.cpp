#include "algebra/basis.h"

#include "framework.h"

TEST(basis_restrict_single_particle) {
  Basis basis = Basis::with_fixed_particle_number(1, 1);

  EXPECT_EQ(basis.set.size(), 2u);
  EXPECT_TRUE(basis.set.contains({Operator::creation(Operator::Spin::Up, 0)}));
  EXPECT_TRUE(basis.set.contains({Operator::creation(Operator::Spin::Down, 0)}));
}

TEST(basis_restrict_two_particles_one_orbital) {
  Basis basis = Basis::with_fixed_particle_number(1, 2);

  EXPECT_EQ(basis.set.size(), 1u);
  EXPECT_TRUE(basis.set.contains(
      {Operator::creation(Operator::Spin::Up, 0), Operator::creation(Operator::Spin::Down, 0)}));
}

TEST(basis_restrict_fixed_spin_sector_single_particle) {
  Basis basis = Basis::with_fixed_particle_number_and_spin(1, 1, 1);

  EXPECT_EQ(basis.set.size(), 1u);
  EXPECT_TRUE(basis.set.contains({Operator::creation(Operator::Spin::Up, 0)}));
}

TEST(basis_restrict_fixed_spin_sector_two_particles) {
  Basis basis = Basis::with_fixed_particle_number_and_spin(2, 2, 0);

  EXPECT_EQ(basis.set.size(), 4u);
  EXPECT_TRUE(basis.set.contains(
      {Operator::creation(Operator::Spin::Up, 0), Operator::creation(Operator::Spin::Down, 0)}));
  EXPECT_TRUE(basis.set.contains(
      {Operator::creation(Operator::Spin::Up, 0), Operator::creation(Operator::Spin::Down, 1)}));
  EXPECT_TRUE(basis.set.contains(
      {Operator::creation(Operator::Spin::Up, 1), Operator::creation(Operator::Spin::Down, 0)}));
  EXPECT_TRUE(basis.set.contains(
      {Operator::creation(Operator::Spin::Up, 1), Operator::creation(Operator::Spin::Down, 1)}));
}

TEST(basis_all_orders_by_size) {
  Basis basis = Basis::with_all_particle_number(1, 2);

  EXPECT_EQ(basis.set.size(), 3u);
  EXPECT_EQ(basis.set[0].size(), 1u);
  EXPECT_EQ(basis.set[1].size(), 1u);
  EXPECT_EQ(basis.set[2].size(), 2u);
  EXPECT_TRUE(basis.set.contains({Operator::creation(Operator::Spin::Up, 0)}));
  EXPECT_TRUE(basis.set.contains({Operator::creation(Operator::Spin::Down, 0)}));
  EXPECT_TRUE(basis.set.contains(
      {Operator::creation(Operator::Spin::Up, 0), Operator::creation(Operator::Spin::Down, 0)}));
}

TEST(basis_fixed_spin_momentum_1d) {
  DynamicIndex index({4});
  DynamicIndex::container_type target_momentum{1};
  Basis basis = Basis::with_fixed_particle_number_spin_momentum(4, 2, 0, index, target_momentum);

  EXPECT_EQ(basis.set.size(), 4u);
  EXPECT_TRUE(basis.set.contains(
      {Operator::creation(Operator::Spin::Up, 0), Operator::creation(Operator::Spin::Down, 1)}));
  EXPECT_TRUE(basis.set.contains(
      {Operator::creation(Operator::Spin::Up, 1), Operator::creation(Operator::Spin::Down, 0)}));
  EXPECT_TRUE(basis.set.contains(
      {Operator::creation(Operator::Spin::Up, 2), Operator::creation(Operator::Spin::Down, 3)}));
  EXPECT_TRUE(basis.set.contains(
      {Operator::creation(Operator::Spin::Up, 3), Operator::creation(Operator::Spin::Down, 2)}));
}

TEST(basis_fixed_spin_momentum_2d_single_particle) {
  DynamicIndex index({2, 2});
  DynamicIndex::container_type target_momentum{1, 0};
  const size_t orbital = index.to_orbital({1, 0});
  Basis basis = Basis::with_fixed_particle_number_spin_momentum(4, 1, 1, index, target_momentum);

  EXPECT_EQ(basis.set.size(), 1u);
  EXPECT_TRUE(basis.set.contains({Operator::creation(Operator::Spin::Up, orbital)}));
}
