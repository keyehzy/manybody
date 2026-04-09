#include <catch2/catch.hpp>
#include <unordered_set>

#include "algebra/operator.h"

namespace {
constexpr std::size_t kMaxValue = FermionOperator::max_index();
}

TEST_CASE("operator_storage_matches_operator_storage") {
  CHECK(sizeof(FermionOperator) == sizeof(OperatorStorage));
}

TEST_CASE("operator_bits_roundtrip") {
  const auto value = kMaxValue / 2;
  FermionOperator op = FermionOperator::creation(FermionOperator::Spin::Down, value);
  CHECK((op.type()) == (FermionOperator::Type::Creation));
  CHECK((op.spin()) == (FermionOperator::Spin::Down));
  CHECK((op.value()) == (value));
}

TEST_CASE("operator_bits_layout") {
  const auto value = kMaxValue;
  FermionOperator op = FermionOperator::annihilation(FermionOperator::Spin::Down, value);
  const auto expected = static_cast<FermionOperator::storage_type>(
      FermionOperator::kTypeBit | FermionOperator::kSpinBit |
      (static_cast<FermionOperator::storage_type>(value) & FermionOperator::kValueMask));
  CHECK((op.data) == (expected));
  CHECK((op.value()) == (value));
}

TEST_CASE("operator_creation_sets_type") {
  FermionOperator op = FermionOperator::creation(FermionOperator::Spin::Up, 9);
  CHECK((op.type()) == (FermionOperator::Type::Creation));
  CHECK((op.spin()) == (FermionOperator::Spin::Up));
  CHECK((op.value()) == (9u));
}

TEST_CASE("operator_annihilation_sets_type") {
  FermionOperator op = FermionOperator::annihilation(FermionOperator::Spin::Down, 5);
  CHECK((op.type()) == (FermionOperator::Type::Annihilation));
  CHECK((op.spin()) == (FermionOperator::Spin::Down));
  CHECK((op.value()) == (5u));
}

TEST_CASE("operator_adjoint") {
  FermionOperator a = FermionOperator::creation(FermionOperator::Spin::Up, 3);
  FermionOperator b = a.adjoint();
  CHECK((b.type()) == (FermionOperator::Type::Annihilation));
  CHECK((b.spin()) == (FermionOperator::Spin::Up));
  CHECK((b.value()) == (3u));
}

TEST_CASE("operator_adjoint_involution") {
  FermionOperator a = FermionOperator::annihilation(FermionOperator::Spin::Down, 7);
  FermionOperator b = a.adjoint().adjoint();
  CHECK((b) == (a));
}

TEST_CASE("operator_flip") {
  FermionOperator a = FermionOperator::creation(FermionOperator::Spin::Up, 4);
  FermionOperator b = a.flip();
  CHECK((b.type()) == (FermionOperator::Type::Creation));
  CHECK((b.spin()) == (FermionOperator::Spin::Down));
  CHECK((b.value()) == (4u));
}

TEST_CASE("operator_flip_involution") {
  FermionOperator a = FermionOperator::annihilation(FermionOperator::Spin::Down, 13);
  FermionOperator b = a.flip().flip();
  CHECK((b) == (a));
}

TEST_CASE("operator_commutes") {
  FermionOperator create_up = FermionOperator::creation(FermionOperator::Spin::Up, 1);
  FermionOperator annihilate_up = FermionOperator::annihilation(FermionOperator::Spin::Up, 1);
  FermionOperator create_down = FermionOperator::creation(FermionOperator::Spin::Down, 1);

  CHECK(!create_up.commutes(annihilate_up));
  CHECK(create_up.commutes(create_down));
}

TEST_CASE("operator_commutes_for_different_value_or_spin") {
  FermionOperator create_up = FermionOperator::creation(FermionOperator::Spin::Up, 1);
  FermionOperator create_up_other_value = FermionOperator::creation(FermionOperator::Spin::Up, 2);
  FermionOperator annihilate_down = FermionOperator::annihilation(FermionOperator::Spin::Down, 1);

  CHECK(create_up.commutes(create_up));
  CHECK(create_up.commutes(create_up_other_value));
  CHECK(create_up.commutes(annihilate_down));
}

TEST_CASE("operator_ordering_type_creation_before_annihilation") {
  FermionOperator create = FermionOperator::creation(FermionOperator::Spin::Up, 1);
  FermionOperator annihilate = FermionOperator::annihilation(FermionOperator::Spin::Up, 1);

  CHECK(create < annihilate);
  CHECK(!(annihilate < create));
}

TEST_CASE("operator_ordering_creation_spin_up_before_down") {
  FermionOperator create_up = FermionOperator::creation(FermionOperator::Spin::Up, 1);
  FermionOperator create_down = FermionOperator::creation(FermionOperator::Spin::Down, 1);

  CHECK(create_up < create_down);
  CHECK(!(create_down < create_up));
}

TEST_CASE("operator_ordering_creation_value_ascending") {
  FermionOperator create_1 = FermionOperator::creation(FermionOperator::Spin::Up, 1);
  FermionOperator create_2 = FermionOperator::creation(FermionOperator::Spin::Up, 2);

  CHECK(create_1 < create_2);
  CHECK(!(create_2 < create_1));
}

TEST_CASE("operator_ordering_annihilation_spin_down_before_up") {
  FermionOperator annihilate_up = FermionOperator::annihilation(FermionOperator::Spin::Up, 1);
  FermionOperator annihilate_down = FermionOperator::annihilation(FermionOperator::Spin::Down, 1);

  CHECK(annihilate_down < annihilate_up);
  CHECK(!(annihilate_up < annihilate_down));
}

TEST_CASE("operator_ordering_annihilation_value_descending") {
  FermionOperator annihilate_1 = FermionOperator::annihilation(FermionOperator::Spin::Up, 1);
  FermionOperator annihilate_2 = FermionOperator::annihilation(FermionOperator::Spin::Up, 2);

  CHECK(annihilate_2 < annihilate_1);
  CHECK(!(annihilate_1 < annihilate_2));
}

TEST_CASE("operator_ordering_equal_operators") {
  FermionOperator op1 = FermionOperator::creation(FermionOperator::Spin::Up, 1);
  FermionOperator op2 = FermionOperator::creation(FermionOperator::Spin::Up, 1);

  CHECK(!(op1 < op2));
  CHECK(!(op2 < op1));
}

TEST_CASE("operator_ordering_normal_order") {
  // Normal ordering: c+(↑,0) c+(↑,1) c+(↓,0) c+(↓,1) c(↓,1) c(↓,0) c(↑,1) c(↑,0)
  FermionOperator c_up_0 = FermionOperator::creation(FermionOperator::Spin::Up, 0);
  FermionOperator c_up_1 = FermionOperator::creation(FermionOperator::Spin::Up, 1);
  FermionOperator c_down_0 = FermionOperator::creation(FermionOperator::Spin::Down, 0);
  FermionOperator c_down_1 = FermionOperator::creation(FermionOperator::Spin::Down, 1);
  FermionOperator a_up_0 = FermionOperator::annihilation(FermionOperator::Spin::Up, 0);
  FermionOperator a_up_1 = FermionOperator::annihilation(FermionOperator::Spin::Up, 1);
  FermionOperator a_down_0 = FermionOperator::annihilation(FermionOperator::Spin::Down, 0);
  FermionOperator a_down_1 = FermionOperator::annihilation(FermionOperator::Spin::Down, 1);

  // Creation operators come first
  CHECK(c_up_0 < a_up_0);
  CHECK(c_down_1 < a_down_1);

  // Creation ordering: spin up before down, then by value ascending
  CHECK(c_up_0 < c_up_1);
  CHECK(c_up_1 < c_down_0);
  CHECK(c_down_0 < c_down_1);

  // Annihilation ordering: spin down before up, then by value descending
  CHECK(a_down_1 < a_down_0);
  CHECK(a_down_0 < a_up_1);
  CHECK(a_up_1 < a_up_0);
}

TEST_CASE("operator_hash_matches_data") {
  FermionOperator op = FermionOperator::annihilation(FermionOperator::Spin::Down, 31);
  std::hash<FermionOperator> hasher;
  CHECK((hasher(op)) == (static_cast<size_t>(op.data)));
}

TEST_CASE("operator_hash_works_in_unordered_set") {
  std::unordered_set<FermionOperator> ops;
  FermionOperator a = FermionOperator::creation(FermionOperator::Spin::Up, 2);
  FermionOperator b = FermionOperator::annihilation(FermionOperator::Spin::Down, 2);

  ops.insert(a);
  ops.insert(b);

  CHECK(ops.find(a) != ops.end());
  CHECK(ops.find(b) != ops.end());
}
