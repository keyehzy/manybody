#pragma once

#include <cstdint>
#include <utility>

#include "algebra/operator.h"
#include "utils/static_vector.h"

using MajoranaIndex = uint16_t;
using MajoranaString = static_vector<MajoranaIndex, 24, uint8_t>;

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

/// Majorana index encoding for a fermionic mode (orbital, spin):
///   flat = 4 * orbital + 2 * parity + spin_bit
/// where parity=0 gives the "even" Majorana (gamma = c + c+)
/// and   parity=1 gives the "odd"  Majorana (gamma = -i(c - c+)).
/// spin_bit: Up=0, Down=1.
///
/// Returns (even_index, odd_index) for the given operator.
inline std::pair<MajoranaIndex, MajoranaIndex> majorana_indices(Operator op) noexcept {
  const auto orbital = static_cast<MajoranaIndex>(op.value());
  const MajoranaIndex spin_bit = (op.spin() == Operator::Spin::Down) ? 1 : 0;
  const MajoranaIndex even_idx = 4 * orbital + spin_bit;
  const MajoranaIndex odd_idx = 4 * orbital + 2 + spin_bit;
  return {even_idx, odd_idx};
}
