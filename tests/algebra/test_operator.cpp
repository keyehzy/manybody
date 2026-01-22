#include <catch2/catch.hpp>
#include <unordered_set>

#include "algebra/operator.h"

TEST_CASE("operator_bits_roundtrip") {
  Operator op = Operator::creation(Operator::Spin::Down, 12);
  CHECK((op.type()) == (Operator::Type::Creation));
  CHECK((op.spin()) == (Operator::Spin::Down));
  CHECK((op.value()) == (12u));
}

TEST_CASE("operator_bits_layout") {
  Operator op = Operator::annihilation(Operator::Spin::Down, 45);
  const Operator::ubyte expected = static_cast<Operator::ubyte>(
      Operator::kTypeBit | Operator::kSpinBit | (45u & Operator::kValueMask));
  CHECK((op.data) == (expected));
  CHECK((op.value()) == (45u));
}

TEST_CASE("operator_creation_sets_type") {
  Operator op = Operator::creation(Operator::Spin::Up, 9);
  CHECK((op.type()) == (Operator::Type::Creation));
  CHECK((op.spin()) == (Operator::Spin::Up));
  CHECK((op.value()) == (9u));
}

TEST_CASE("operator_annihilation_sets_type") {
  Operator op = Operator::annihilation(Operator::Spin::Down, 5);
  CHECK((op.type()) == (Operator::Type::Annihilation));
  CHECK((op.spin()) == (Operator::Spin::Down));
  CHECK((op.value()) == (5u));
}

TEST_CASE("operator_adjoint") {
  Operator a = Operator::creation(Operator::Spin::Up, 3);
  Operator b = a.adjoint();
  CHECK((b.type()) == (Operator::Type::Annihilation));
  CHECK((b.spin()) == (Operator::Spin::Up));
  CHECK((b.value()) == (3u));
}

TEST_CASE("operator_adjoint_involution") {
  Operator a = Operator::annihilation(Operator::Spin::Down, 7);
  Operator b = a.adjoint().adjoint();
  CHECK((b) == (a));
}

TEST_CASE("operator_flip") {
  Operator a = Operator::creation(Operator::Spin::Up, 4);
  Operator b = a.flip();
  CHECK((b.type()) == (Operator::Type::Creation));
  CHECK((b.spin()) == (Operator::Spin::Down));
  CHECK((b.value()) == (4u));
}

TEST_CASE("operator_flip_involution") {
  Operator a = Operator::annihilation(Operator::Spin::Down, 13);
  Operator b = a.flip().flip();
  CHECK((b) == (a));
}

TEST_CASE("operator_commutes") {
  Operator create_up = Operator::creation(Operator::Spin::Up, 1);
  Operator annihilate_up = Operator::annihilation(Operator::Spin::Up, 1);
  Operator create_down = Operator::creation(Operator::Spin::Down, 1);

  CHECK(!create_up.commutes(annihilate_up));
  CHECK(create_up.commutes(create_down));
}

TEST_CASE("operator_commutes_for_different_value_or_spin") {
  Operator create_up = Operator::creation(Operator::Spin::Up, 1);
  Operator create_up_other_value = Operator::creation(Operator::Spin::Up, 2);
  Operator annihilate_down = Operator::annihilation(Operator::Spin::Down, 1);

  CHECK(create_up.commutes(create_up));
  CHECK(create_up.commutes(create_up_other_value));
  CHECK(create_up.commutes(annihilate_down));
}

TEST_CASE("operator_ordering_type_creation_before_annihilation") {
  Operator create = Operator::creation(Operator::Spin::Up, 1);
  Operator annihilate = Operator::annihilation(Operator::Spin::Up, 1);

  CHECK(create < annihilate);
  CHECK(!(annihilate < create));
}

TEST_CASE("operator_ordering_creation_spin_up_before_down") {
  Operator create_up = Operator::creation(Operator::Spin::Up, 1);
  Operator create_down = Operator::creation(Operator::Spin::Down, 1);

  CHECK(create_up < create_down);
  CHECK(!(create_down < create_up));
}

TEST_CASE("operator_ordering_creation_value_ascending") {
  Operator create_1 = Operator::creation(Operator::Spin::Up, 1);
  Operator create_2 = Operator::creation(Operator::Spin::Up, 2);

  CHECK(create_1 < create_2);
  CHECK(!(create_2 < create_1));
}

TEST_CASE("operator_ordering_annihilation_spin_down_before_up") {
  Operator annihilate_up = Operator::annihilation(Operator::Spin::Up, 1);
  Operator annihilate_down = Operator::annihilation(Operator::Spin::Down, 1);

  CHECK(annihilate_down < annihilate_up);
  CHECK(!(annihilate_up < annihilate_down));
}

TEST_CASE("operator_ordering_annihilation_value_descending") {
  Operator annihilate_1 = Operator::annihilation(Operator::Spin::Up, 1);
  Operator annihilate_2 = Operator::annihilation(Operator::Spin::Up, 2);

  CHECK(annihilate_2 < annihilate_1);
  CHECK(!(annihilate_1 < annihilate_2));
}

TEST_CASE("operator_ordering_equal_operators") {
  Operator op1 = Operator::creation(Operator::Spin::Up, 1);
  Operator op2 = Operator::creation(Operator::Spin::Up, 1);

  CHECK(!(op1 < op2));
  CHECK(!(op2 < op1));
}

TEST_CASE("operator_ordering_normal_order") {
  // Normal ordering: c+(↑,0) c+(↑,1) c+(↓,0) c+(↓,1) c(↓,1) c(↓,0) c(↑,1) c(↑,0)
  Operator c_up_0 = Operator::creation(Operator::Spin::Up, 0);
  Operator c_up_1 = Operator::creation(Operator::Spin::Up, 1);
  Operator c_down_0 = Operator::creation(Operator::Spin::Down, 0);
  Operator c_down_1 = Operator::creation(Operator::Spin::Down, 1);
  Operator a_up_0 = Operator::annihilation(Operator::Spin::Up, 0);
  Operator a_up_1 = Operator::annihilation(Operator::Spin::Up, 1);
  Operator a_down_0 = Operator::annihilation(Operator::Spin::Down, 0);
  Operator a_down_1 = Operator::annihilation(Operator::Spin::Down, 1);

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
  Operator op = Operator::annihilation(Operator::Spin::Down, 31);
  std::hash<Operator> hasher;
  CHECK((hasher(op)) == (static_cast<size_t>(op.data)));
}

TEST_CASE("operator_hash_works_in_unordered_set") {
  std::unordered_set<Operator> ops;
  Operator a = Operator::creation(Operator::Spin::Up, 2);
  Operator b = Operator::annihilation(Operator::Spin::Down, 2);

  ops.insert(a);
  ops.insert(b);

  CHECK(ops.find(a) != ops.end());
  CHECK(ops.find(b) != ops.end());
}
