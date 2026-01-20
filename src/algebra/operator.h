#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <sstream>
#include <string>

struct Operator {
  using ubyte = uint8_t;

  static constexpr ubyte kTypeShift = 7;
  static constexpr ubyte kSpinShift = 6;
  static constexpr ubyte kValueMask = 0x3F;

  static constexpr ubyte kTypeBit = 1 << kTypeShift;
  static constexpr ubyte kSpinBit = 1 << kSpinShift;

  enum class Type : ubyte { Creation = 0, Annihilation = 1 };
  enum class Spin : ubyte { Up = 0, Down = 1 };

  constexpr Operator() noexcept = default;

  constexpr explicit Operator(Type type, Spin spin, size_t value) noexcept
      : data(static_cast<ubyte>((static_cast<ubyte>(type) << kTypeShift) |
                                (static_cast<ubyte>(spin) << kSpinShift) |
                                (static_cast<ubyte>(value) & kValueMask))) {
    assert(value <= kValueMask);
  }

  constexpr Type type() const noexcept { return static_cast<Type>(data >> kTypeShift); }

  constexpr Spin spin() const noexcept { return static_cast<Spin>((data >> kSpinShift) & 1); }

  constexpr size_t value() const noexcept { return data & kValueMask; }

  void to_string(std::ostringstream& oss) const {
    const char* spin_arrow = spin() == Spin::Up ? "↑" : "↓";
    oss << "c";
    if (type() == Type::Creation) {
      oss << "+";
    }
    oss << "(" << spin_arrow << ", " << value() << ")";
  }

  std::string to_string() const {
    std::ostringstream oss;
    to_string(oss);
    return oss.str();
  }

  constexpr bool operator<(Operator other) const noexcept {
    bool is_creation = other.type() == Type::Creation;
    if (type() != other.type()) {
      return type() < other.type();
    } else if (spin() != other.spin()) {
      return is_creation ? spin() < other.spin() : spin() > other.spin();
    } else {
      return is_creation ? value() < other.value() : value() > other.value();
    }
  }
  constexpr bool operator==(Operator other) const noexcept { return data == other.data; }

  constexpr Operator adjoint() const noexcept { return Operator(data ^ kTypeBit); }

  constexpr Operator flip() const noexcept { return Operator(data ^ kSpinBit); }

  constexpr bool commutes(Operator other) const noexcept { return (data ^ other.data) != kTypeBit; }

  constexpr explicit Operator(ubyte x) noexcept : data(x) {}

  constexpr static Operator creation(Spin spin, size_t value) noexcept {
    return Operator(Type::Creation, spin, value);
  }

  constexpr static Operator annihilation(Spin spin, size_t value) noexcept {
    return Operator(Type::Annihilation, spin, value);
  }

  constexpr static size_t max_index() noexcept { return kValueMask; }

  ubyte data{};
};
static_assert(sizeof(Operator) == 1);

template <>
struct std::hash<Operator> {
  [[nodiscard]] constexpr std::size_t operator()(Operator op) const noexcept { return op.data; }
};
