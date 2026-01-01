#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>

struct Operator {
  using ubyte = uint8_t;

  static constexpr ubyte kTypeShift = 7;
  static constexpr ubyte kSpinShift = 6;
  static constexpr ubyte kValueMask = 0x3F;

  static constexpr ubyte kTypeBit = 1 << kTypeShift;
  static constexpr ubyte kSpinBit = 1 << kSpinShift;

  enum class Type : ubyte { Creation = 0, Annihilation = 1 };
  enum class Spin : ubyte { Up = 0, Down = 1 };

  constexpr explicit Operator(Type type, Spin spin, size_t value) noexcept
      : data(static_cast<ubyte>((static_cast<ubyte>(type) << kTypeShift) |
                                (static_cast<ubyte>(spin) << kSpinShift) |
                                (static_cast<ubyte>(value) & kValueMask))) {
    assert(value <= kValueMask);
  }

  constexpr Type type() const noexcept {
    return static_cast<Type>(data >> kTypeShift);
  }

  constexpr Spin spin() const noexcept {
    return static_cast<Spin>((data >> kSpinShift) & 1);
  }

  constexpr size_t value() const noexcept { return data & kValueMask; }

  constexpr bool operator<(Operator other) const noexcept {
    return data < other.data;
  }
  constexpr bool operator==(Operator other) const noexcept {
    return data == other.data;
  }

  constexpr Operator adjoint() const noexcept {
    return Operator(data ^ kTypeBit);
  }

  constexpr bool commutes(Operator other) const noexcept {
    return (data ^ other.data) != kTypeBit;
  }

  constexpr explicit Operator(ubyte x) noexcept : data(x) {}

  ubyte data{};
};
