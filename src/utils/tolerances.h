#pragma once

#include <limits>

namespace tolerances {

template <typename T>
constexpr T tolerance() {
  return static_cast<T>(1000) * std::numeric_limits<T>::epsilon();
}

}  // namespace tolerances
