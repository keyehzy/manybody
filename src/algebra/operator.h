#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <sstream>
#include <string>
#include <type_traits>

using OperatorStorage = std::uint8_t;

template <class Storage>
struct BasicOperator {
  using storage_type = std::make_unsigned_t<Storage>;
  using ubyte = storage_type;

  static_assert(std::is_integral_v<storage_type>, "Storage must be an integral type.");
  static_assert(std::is_unsigned_v<storage_type>,
                "Storage must be unsigned (or convertible to unsigned).");

  static constexpr unsigned kBits = std::numeric_limits<storage_type>::digits;
  static_assert(kBits >= 3, "Need at least 3 bits: type, spin, and >=1 value bit.");

  static constexpr unsigned kTypeShift = kBits - 1;
  static constexpr unsigned kSpinShift = kBits - 2;

  static constexpr storage_type kTypeBit = (storage_type{1} << kTypeShift);
  static constexpr storage_type kSpinBit = (storage_type{1} << kSpinShift);

  static constexpr storage_type kValueMask = (storage_type{1} << kSpinShift) - storage_type{1};

  enum class Type : storage_type { Creation = 0, Annihilation = 1 };
  enum class Spin : storage_type { Up = 0, Down = 1 };

  constexpr BasicOperator() noexcept = default;

  constexpr explicit BasicOperator(Type type, Spin spin, std::size_t value) noexcept
      : data(pack(type, spin, value)) {
    assert(value <= max_index());
  }

  constexpr Type type() const noexcept { return static_cast<Type>(data >> kTypeShift); }

  constexpr Spin spin() const noexcept {
    return static_cast<Spin>((data >> kSpinShift) & storage_type{1});
  }

  constexpr std::size_t value() const noexcept {
    return static_cast<std::size_t>(data & kValueMask);
  }

  void to_string(std::ostringstream& oss) const {
    const char* spin_arrow = spin() == Spin::Up ? "↑" : "↓";
    oss << "c";
    if (type() == Type::Creation) oss << "+";
    oss << "(" << spin_arrow << ", " << value() << ")";
  }

  std::string to_string() const {
    std::ostringstream oss;
    to_string(oss);
    return oss.str();
  }

  constexpr bool operator<(BasicOperator other) const noexcept {
    const bool other_is_creation = (other.type() == Type::Creation);

    if (type() != other.type()) {
      return type() < other.type();
    } else if (spin() != other.spin()) {
      return other_is_creation ? (spin() < other.spin()) : (spin() > other.spin());
    } else {
      return other_is_creation ? (value() < other.value()) : (value() > other.value());
    }
  }

  constexpr bool operator==(BasicOperator other) const noexcept { return data == other.data; }

  constexpr BasicOperator adjoint() const noexcept { return BasicOperator(data ^ kTypeBit); }
  constexpr BasicOperator flip() const noexcept { return BasicOperator(data ^ kSpinBit); }

  constexpr bool commutes(BasicOperator other) const noexcept {
    return (data ^ other.data) != kTypeBit;
  }

  constexpr explicit BasicOperator(storage_type raw) noexcept : data(raw) {}

  constexpr static BasicOperator creation(Spin spin, std::size_t value) noexcept {
    return BasicOperator(Type::Creation, spin, value);
  }

  constexpr static BasicOperator annihilation(Spin spin, std::size_t value) noexcept {
    return BasicOperator(Type::Annihilation, spin, value);
  }

  constexpr static std::size_t max_index() noexcept { return static_cast<std::size_t>(kValueMask); }

  storage_type data{};

 private:
  static constexpr storage_type pack(Type type, Spin spin, std::size_t value) noexcept {
    return (static_cast<storage_type>(type) << kTypeShift) |
           (static_cast<storage_type>(spin) << kSpinShift) |
           (static_cast<storage_type>(value) & kValueMask);
  }
};

using Operator = BasicOperator<OperatorStorage>;

static_assert(sizeof(Operator) == sizeof(OperatorStorage));

template <>
struct std::hash<Operator> {
  [[nodiscard]] constexpr std::size_t operator()(Operator op) const noexcept {
    return static_cast<std::size_t>(op.data);
  }
};
