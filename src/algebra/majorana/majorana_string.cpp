#include "algebra/majorana/majorana_string.h"

namespace majorana {

void to_string(std::ostringstream& oss, const MajoranaString& str) {
  for (const auto& op : str) {
    op.to_string(oss);
  }
}

std::string to_string(const MajoranaString& str) {
  std::ostringstream oss;
  to_string(oss, str);
  return oss.str();
}

MajoranaProduct canonicalize(const MajoranaString& str) noexcept {
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

MajoranaProduct multiply_strings(const MajoranaString& a, const MajoranaString& b) noexcept {
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

}  // namespace majorana
