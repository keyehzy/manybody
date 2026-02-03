#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <sstream>
#include <string>
#include <type_traits>

#include "algebra/operator.h"

namespace majorana {

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

  // Printing convention: gamma(index, spin) with index = 2 * orbital + parity,
  // where parity is 0 for even (gamma_e) and 1 for odd (gamma_o).
  // This is the 0-based equivalent of the common paper convention:
  //   gamma_{2j-1, sigma} = c_{j sigma} + c^\dagger_{j sigma}
  //   gamma_{2j,   sigma} = -i(c_{j sigma} - c^\dagger_{j sigma}).
  void to_string(std::ostringstream& oss) const {
    const char* spin_arrow = spin() == Operator::Spin::Up ? "↑" : "↓";
    const std::size_t index = 2 * orbital() + (is_odd() ? 1u : 0u);
    oss << "γ(" << index << ", " << spin_arrow << ")";
  }

  std::string to_string() const {
    std::ostringstream oss;
    to_string(oss);
    return oss.str();
  }

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

}  // namespace majorana
