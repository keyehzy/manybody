#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace utils {

inline size_t canonicalize_momentum(int64_t value, size_t lattice_size) {
  if (lattice_size == 0) {
    return 0;
  }
  const int64_t L = static_cast<int64_t>(lattice_size);
  int64_t mod = value % L;
  if (mod < 0) {
    mod += L;
  }
  return static_cast<size_t>(mod);
}

inline std::vector<size_t> canonicalize_momentum(const std::vector<int64_t>& momentum,
                                                 const std::vector<size_t>& size) {
  std::vector<size_t> result(momentum.size());
  for (size_t d = 0; d < momentum.size(); ++d) {
    result[d] = canonicalize_momentum(momentum[d], size[d]);
  }
  return result;
}

}  // namespace utils
