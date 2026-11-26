#include <catch2/catch.hpp>

#include "algebra/fermion/basis.h"

TEST_CASE("basis_restrict_single_particle") {
  FermionBasis basis = FermionBasis::with_fixed_particle_number(1, 1);

  CHECK((basis.set.size()) == (2u));
  CHECK(basis.set.contains({FermionOperator::creation(FermionOperator::Spin::Up, 0)}));
  CHECK(basis.set.contains({FermionOperator::creation(FermionOperator::Spin::Down, 0)}));
}

TEST_CASE("basis_restrict_two_particles_one_orbital") {
  FermionBasis basis = FermionBasis::with_fixed_particle_number(1, 2);

  CHECK((basis.set.size()) == (1u));
  CHECK(basis.set.contains({FermionOperator::creation(FermionOperator::Spin::Up, 0),
                            FermionOperator::creation(FermionOperator::Spin::Down, 0)}));
}

TEST_CASE("basis_restrict_fixed_spin_sector_single_particle") {
  FermionBasis basis = FermionBasis::with_fixed_particle_number_and_spin(1, 1, 1);

  CHECK((basis.set.size()) == (1u));
  CHECK(basis.set.contains({FermionOperator::creation(FermionOperator::Spin::Up, 0)}));
}

TEST_CASE("basis_restrict_fixed_spin_sector_two_particles") {
  FermionBasis basis = FermionBasis::with_fixed_particle_number_and_spin(2, 2, 0);

  CHECK((basis.set.size()) == (4u));
  CHECK(basis.set.contains({FermionOperator::creation(FermionOperator::Spin::Up, 0),
                            FermionOperator::creation(FermionOperator::Spin::Down, 0)}));
  CHECK(basis.set.contains({FermionOperator::creation(FermionOperator::Spin::Up, 0),
                            FermionOperator::creation(FermionOperator::Spin::Down, 1)}));
  CHECK(basis.set.contains({FermionOperator::creation(FermionOperator::Spin::Up, 1),
                            FermionOperator::creation(FermionOperator::Spin::Down, 0)}));
  CHECK(basis.set.contains({FermionOperator::creation(FermionOperator::Spin::Up, 1),
                            FermionOperator::creation(FermionOperator::Spin::Down, 1)}));
}

TEST_CASE("basis_all_orders_by_size") {
  FermionBasis basis = FermionBasis::with_all_particle_number(1, 2);

  CHECK((basis.set.size()) == (3u));
  CHECK((basis.set[0].size()) == (1u));
  CHECK((basis.set[1].size()) == (1u));
  CHECK((basis.set[2].size()) == (2u));
  CHECK(basis.set.contains({FermionOperator::creation(FermionOperator::Spin::Up, 0)}));
  CHECK(basis.set.contains({FermionOperator::creation(FermionOperator::Spin::Down, 0)}));
  CHECK(basis.set.contains({FermionOperator::creation(FermionOperator::Spin::Up, 0),
                            FermionOperator::creation(FermionOperator::Spin::Down, 0)}));
}

TEST_CASE("basis_fixed_spin_momentum_1d") {
  Index index({4});
  Index::container_type target_momentum{1};
  FermionBasis basis =
      FermionBasis::with_fixed_particle_number_spin_momentum(4, 2, 0, index, target_momentum);

  CHECK((basis.set.size()) == (4u));
  CHECK(basis.set.contains({FermionOperator::creation(FermionOperator::Spin::Up, 0),
                            FermionOperator::creation(FermionOperator::Spin::Down, 1)}));
  CHECK(basis.set.contains({FermionOperator::creation(FermionOperator::Spin::Up, 1),
                            FermionOperator::creation(FermionOperator::Spin::Down, 0)}));
  CHECK(basis.set.contains({FermionOperator::creation(FermionOperator::Spin::Up, 2),
                            FermionOperator::creation(FermionOperator::Spin::Down, 3)}));
  CHECK(basis.set.contains({FermionOperator::creation(FermionOperator::Spin::Up, 3),
                            FermionOperator::creation(FermionOperator::Spin::Down, 2)}));
}

TEST_CASE("basis_fixed_spin_momentum_2d_single_particle") {
  Index index({2, 2});
  Index::container_type target_momentum{1, 0};
  const size_t orbital = index({1, 0});
  FermionBasis basis =
      FermionBasis::with_fixed_particle_number_spin_momentum(4, 1, 1, index, target_momentum);

  CHECK((basis.set.size()) == (1u));
  CHECK(basis.set.contains({FermionOperator::creation(FermionOperator::Spin::Up, orbital)}));
}

TEST_CASE("basis_factories_reject_invalid_arguments") {
  // These checks used to be assert-only and were silently skipped in release builds.
  SECTION("too many particles for the spin-orbitals") {
    CHECK_THROWS_AS(FermionBasis::with_fixed_particle_number(2, 5), std::invalid_argument);
    CHECK_THROWS_AS(FermionBasis::with_all_particle_number(2, 5), std::invalid_argument);
  }
  SECTION("more particles than a term can hold") {
    const size_t too_many = FermionBasis::key_type::max_size() + 1;
    CHECK_THROWS_AS(FermionBasis::with_fixed_particle_number(too_many, too_many),
                    std::invalid_argument);
  }
  SECTION("spin projection inconsistent with particle number") {
    CHECK_THROWS_AS(FermionBasis::with_fixed_particle_number_and_spin(4, 2, 4),
                    std::invalid_argument);  // |2Sz| > N
    CHECK_THROWS_AS(FermionBasis::with_fixed_particle_number_and_spin(4, 3, 0),
                    std::invalid_argument);  // parity mismatch
    CHECK_THROWS_AS(FermionBasis::with_fixed_particle_number_and_spin(2, 4, 2),
                    std::invalid_argument);  // 3 up electrons on 2 orbitals
  }
  SECTION("momentum sector inconsistent with the lattice") {
    Index index({4});
    CHECK_THROWS_AS(FermionBasis::with_fixed_particle_number_spin_momentum(
                        4, 2, 0, index, Index::container_type{4}),
                    std::invalid_argument);  // momentum out of range
    CHECK_THROWS_AS(FermionBasis::with_fixed_particle_number_spin_momentum(
                        4, 2, 0, index, Index::container_type{1, 0}),
                    std::invalid_argument);  // wrong number of components
    CHECK_THROWS_AS(FermionBasis::with_fixed_particle_number_spin_momentum(
                        5, 2, 0, index, Index::container_type{1}),
                    std::invalid_argument);  // index size != orbitals
  }
}
