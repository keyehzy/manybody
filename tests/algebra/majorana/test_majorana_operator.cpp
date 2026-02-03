#include <catch2/catch.hpp>
#include <cstdint>

#include "algebra/majorana/majorana_operator.h"

using namespace majorana;

namespace {
constexpr std::size_t kMaxMajoranaOrbital = MajoranaOperator::max_index();
}

TEST_CASE("majorana_operator_storage_matches_majorana_storage") {
  CHECK(sizeof(MajoranaOperator) == sizeof(MajoranaOperatorStorage));
}

TEST_CASE("majorana_operator_bits_roundtrip") {
  const auto orbital = kMaxMajoranaOrbital / 2;
  MajoranaOperator op = MajoranaOperator::odd(orbital, Operator::Spin::Down);

  CHECK(op.orbital() == orbital);
  CHECK(op.spin() == Operator::Spin::Down);
  CHECK(op.parity() == MajoranaOperator::Parity::Odd);
}

TEST_CASE("majorana_operator_bits_layout") {
  const auto orbital = kMaxMajoranaOrbital;
  MajoranaOperator op = MajoranaOperator::odd(orbital, Operator::Spin::Down);
  const auto expected = static_cast<MajoranaOperator::storage_type>(
      (static_cast<MajoranaOperator::storage_type>(orbital) << MajoranaOperator::kOrbitalShift) |
      MajoranaOperator::kParityBit | MajoranaOperator::kSpinBit);

  CHECK(op.data == expected);
}

TEST_CASE("basic_majorana_operator_supports_small_storage") {
  using SmallMajorana = BasicMajoranaOperator<std::uint8_t>;
  const auto orbital = SmallMajorana::max_index();
  SmallMajorana op = SmallMajorana::even(orbital, Operator::Spin::Up);

  CHECK(sizeof(SmallMajorana) == sizeof(std::uint8_t));
  CHECK(op.orbital() == orbital);
  CHECK(op.spin() == Operator::Spin::Up);
  CHECK(op.parity() == SmallMajorana::Parity::Even);
}
