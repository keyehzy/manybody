#pragma once

#include <cstddef>
#include <vector>

template <class Container>
static void stable_sort_with_parity(Container& a, Container& tmp, std::size_t l, std::size_t r,
                                    bool& odd) noexcept {
  std::size_t n = r - l;
  if (n <= 1) return;

  std::size_t m = l + n / 2;
  stable_sort_with_parity(a, tmp, l, m, odd);
  stable_sort_with_parity(a, tmp, m, r, odd);

  // Merge a[l:m) and a[m:r) into tmp[l:r), stable.
  std::size_t i = l;
  std::size_t j = m;
  std::size_t k = l;

  while (i < m && j < r) {
    // Stable: if a[j] is not strictly less than a[i], take left.
    if (!(a[j] < a[i])) {
      tmp[k++] = a[i++];
    } else {
      // Taking from right before remaining left contributes (m - i) inversions.
      std::size_t passed = (m - i);
      if (passed & 1u) odd = !odd;
      tmp[k++] = a[j++];
    }
  }

  while (i < m) tmp[k++] = a[i++];
  while (j < r) tmp[k++] = a[j++];

  for (std::size_t t = l; t < r; ++t) {
    a[t] = tmp[t];
  }
}
