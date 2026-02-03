#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

#include "algebra/operator.h"

using MajoranaOperatorStorage = std::uint16_t;

template <class Storage>
struct BasicMajoranaOperator {
  using storage_type = std::make_unsigned_t<Storage>;

  static_assert(std::is_integral_v<storage_type>, "Storage must be an integral type.");
  static_assert(std::is_unsigned_v<storage_type>,
                "Storage must be unsigned (or convertible to unsigned).");

  static constexpr unsigned kBits = std::numeric_limits<storage_type>::digits;
  static_assert(kBits >= 3, "Need at least 3 bits: orbital, parity, and spin.");

  static constexpr unsigned kSpinShift = 0;
  static constexpr unsigned kParityShift = 1;
  static constexpr unsigned kOrbitalShift = 2;

  static constexpr storage_type kSpinBit = storage_type{1} << kSpinShift;
  static constexpr storage_type kParityBit = storage_type{1} << kParityShift;
  static constexpr storage_type kOrbitalMask =
      (storage_type{1} << (kBits - kOrbitalShift)) - storage_type{1};

  enum class Parity : storage_type { Even = 0, Odd = 1 };

  constexpr BasicMajoranaOperator() noexcept = default;

  constexpr explicit BasicMajoranaOperator(std::size_t orbital, Operator::Spin spin,
                                           Parity parity) noexcept
      : data(pack(orbital, spin, parity)) {
    assert(orbital <= max_index());
  }

  constexpr std::size_t orbital() const noexcept {
    return static_cast<std::size_t>(data >> kOrbitalShift);
  }

  constexpr Operator::Spin spin() const noexcept {
    return ((data & kSpinBit) == storage_type{0}) ? Operator::Spin::Up : Operator::Spin::Down;
  }

  constexpr Parity parity() const noexcept {
    return ((data >> kParityShift) & storage_type{1}) == storage_type{0} ? Parity::Even
                                                                         : Parity::Odd;
  }

  constexpr bool is_even() const noexcept { return parity() == Parity::Even; }
  constexpr bool is_odd() const noexcept { return parity() == Parity::Odd; }

  constexpr bool operator<(BasicMajoranaOperator other) const noexcept { return data < other.data; }
  constexpr bool operator>(BasicMajoranaOperator other) const noexcept { return data > other.data; }
  constexpr bool operator==(BasicMajoranaOperator other) const noexcept {
    return data == other.data;
  }
  constexpr bool operator!=(BasicMajoranaOperator other) const noexcept {
    return data != other.data;
  }

  constexpr static BasicMajoranaOperator even(std::size_t orbital, Operator::Spin spin) noexcept {
    return BasicMajoranaOperator(orbital, spin, Parity::Even);
  }

  constexpr static BasicMajoranaOperator odd(std::size_t orbital, Operator::Spin spin) noexcept {
    return BasicMajoranaOperator(orbital, spin, Parity::Odd);
  }

  constexpr static std::size_t max_index() noexcept {
    return static_cast<std::size_t>(kOrbitalMask);
  }

  storage_type data{};

 private:
  static constexpr storage_type pack(std::size_t orbital, Operator::Spin spin,
                                     Parity parity) noexcept {
    return ((static_cast<storage_type>(orbital) & kOrbitalMask) << kOrbitalShift) |
           (static_cast<storage_type>(parity) << kParityShift) |
           (static_cast<storage_type>(spin) << kSpinShift);
  }
};

using MajoranaOperator = BasicMajoranaOperator<MajoranaOperatorStorage>;

static_assert(sizeof(MajoranaOperator) == sizeof(MajoranaOperatorStorage));
