#pragma once

#include <cstddef>
#include <cstdint>
#include <sstream>
#include <string>

#include "algebra/majorana/majorana_operator.h"
#include "utils/static_vector.h"

using MajoranaString = static_vector<MajoranaOperator, 24, std::uint8_t>;

namespace majorana_string {
inline void to_string(std::ostringstream& oss, const MajoranaString& str) {
  for (const auto& op : str) {
    op.to_string(oss);
  }
}

inline std::string to_string(const MajoranaString& str) {
  std::ostringstream oss;
  to_string(oss, str);
  return oss.str();
}
}  // namespace majorana_string

struct MajoranaProduct {
  int sign = 1;
  MajoranaString string{};
};

/// Multiply two sorted Majorana strings using the Clifford algebra relation
/// {gamma_i, gamma_j} = 2 * delta_ij.  Each pair of equal indices cancels
/// (gamma_i^2 = 1) and every anticommutation swap contributes a sign flip.
inline constexpr MajoranaProduct multiply_strings(const MajoranaString& a,
                                                  const MajoranaString& b) noexcept {
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

namespace majorana_string {
inline MajoranaProduct canonicalize(const MajoranaString& str) noexcept {
  MajoranaProduct result;
  result.sign = 1;

  for (const auto& op : str) {
    MajoranaString single;
    single.push_back(op);
    auto product = multiply_strings(result.string, single);
    result.sign *= product.sign;
    result.string = product.string;
  }

  return result;
}
}  // namespace majorana_string
