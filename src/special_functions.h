#pragma once

#ifndef __STDCPP_WANT_MATH_SPEC_FUNCS__
#define __STDCPP_WANT_MATH_SPEC_FUNCS__ 1
#endif

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <type_traits>

#if defined(__cpp_lib_math_special_functions) ||                                     \
    (defined(__STDCPP_MATH_SPEC_FUNCS__) && __STDCPP_MATH_SPEC_FUNCS__ >= 201003L && \
     defined(__STDCPP_WANT_MATH_SPEC_FUNCS__))
#define MANYBODY_HAS_STD_BESSEL_I 1
#else
#define MANYBODY_HAS_STD_BESSEL_I 0
#endif

template <typename RealType>
inline RealType bessel_i_series(size_t order, RealType x, RealType tolerance) {
  const RealType abs_x = std::abs(x);
  if (abs_x == static_cast<RealType>(0)) {
    return order == 0 ? static_cast<RealType>(1) : static_cast<RealType>(0);
  }

  const RealType half_x = abs_x / static_cast<RealType>(2);
  RealType term = static_cast<RealType>(1);
  for (size_t k = 1; k <= order; ++k) {
    term *= half_x / static_cast<RealType>(k);
  }
  RealType sum = term;
  const RealType eps =
      std::max(tolerance * static_cast<RealType>(0.1), std::numeric_limits<RealType>::epsilon());
  for (size_t m = 1; m < 200; ++m) {
    term *= (half_x * half_x) / static_cast<RealType>(m * (m + order));
    sum += term;
    if (term < eps * sum) {
      break;
    }
  }
  return sum;
}

template <typename RealType>
inline RealType bessel_i(size_t order, RealType x, RealType tolerance) {
  const RealType abs_x = std::abs(x);
#if MANYBODY_HAS_STD_BESSEL_I
  (void)tolerance;
  return std::cyl_bessel_i(static_cast<RealType>(order), abs_x);
#else
  return bessel_i_series(order, abs_x, tolerance);
#endif
}
