#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>

#include "algebra/operator.h"
#include "utils/static_vector.h"

struct MajoranaElement {
  using storage_type = std::uint16_t;

  enum class Parity : storage_type { Even = 0, Odd = 1 };

  constexpr MajoranaElement() noexcept = default;

  constexpr MajoranaElement(std::size_t orbital, Operator::Spin spin, Parity parity) noexcept
      : data(pack(orbital, spin, parity)) {}

  constexpr std::size_t orbital() const noexcept { return static_cast<std::size_t>(data >> 2); }

  constexpr Operator::Spin spin() const noexcept {
    return ((data & storage_type{1}) == storage_type{0}) ? Operator::Spin::Up
                                                         : Operator::Spin::Down;
  }

  constexpr Parity parity() const noexcept {
    return ((data >> 1) & storage_type{1}) == storage_type{0} ? Parity::Even : Parity::Odd;
  }

  constexpr bool is_even() const noexcept { return parity() == Parity::Even; }
  constexpr bool is_odd() const noexcept { return parity() == Parity::Odd; }

  constexpr bool operator<(MajoranaElement other) const noexcept { return data < other.data; }
  constexpr bool operator>(MajoranaElement other) const noexcept { return data > other.data; }
  constexpr bool operator==(MajoranaElement other) const noexcept { return data == other.data; }
  constexpr bool operator!=(MajoranaElement other) const noexcept { return data != other.data; }

  constexpr static MajoranaElement even(std::size_t orbital, Operator::Spin spin) noexcept {
    return MajoranaElement(orbital, spin, Parity::Even);
  }

  constexpr static MajoranaElement odd(std::size_t orbital, Operator::Spin spin) noexcept {
    return MajoranaElement(orbital, spin, Parity::Odd);
  }

 private:
  static constexpr storage_type pack(std::size_t orbital, Operator::Spin spin,
                                     Parity parity) noexcept {
    assert(orbital <= (std::numeric_limits<storage_type>::max() - 3u) / 4u);
    return static_cast<storage_type>(4u * orbital + 2u * static_cast<storage_type>(parity) +
                                     static_cast<storage_type>(spin));
  }

  storage_type data{};
};

using MajoranaString = static_vector<MajoranaElement, 24, std::uint8_t>;

struct MajoranaProduct {
  int sign = 1;
  MajoranaString string{};
};

/// Multiply two sorted Majorana strings using the Clifford algebra relation
/// {gamma_i, gamma_j} = 2 * delta_ij.  Each pair of equal indices cancels
/// (gamma_i^2 = 1) and every anticommutation swap contributes a sign flip.
inline MajoranaProduct multiply_strings(const MajoranaString& a, const MajoranaString& b) noexcept {
  MajoranaProduct result;
  result.sign = 1;

  // Merge-sort style: walk both sorted strings simultaneously.
  // Count how many elements from b must pass elements from a (= swaps).
  size_t i = 0;
  size_t j = 0;
  const size_t na = a.size();
  const size_t nb = b.size();

  while (i < na && j < nb) {
    if (a[i] < b[j]) {
      result.string.push_back(a[i]);
      ++i;
    } else if (a[i] > b[j]) {
      // b[j] must hop past (na - i) remaining elements of a in the merged
      // string.  But we only need parity of the number of swaps past the
      // *surviving* elements already placed, so just count the remaining a
      // elements it passes.
      if ((na - i) % 2 != 0) {
        result.sign = -result.sign;
      }
      result.string.push_back(b[j]);
      ++j;
    } else {
      // Equal indices: gamma_i * gamma_i = 1 (cancel the pair).
      // b[j] had to hop past (na - i - 1) remaining a-elements to reach its
      // partner a[i], then the pair annihilates.
      if ((na - i - 1) % 2 != 0) {
        result.sign = -result.sign;
      }
      ++i;
      ++j;
    }
  }

  // Append remaining elements from whichever string is not exhausted.
  while (i < na) {
    result.string.push_back(a[i]);
    ++i;
  }
  while (j < nb) {
    result.string.push_back(b[j]);
    ++j;
  }

  return result;
}
