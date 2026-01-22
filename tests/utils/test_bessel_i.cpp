#include <catch2/catch.hpp>
#include <cmath>

#include "utils/special_functions.h"

TEST_CASE("bessel_i_handles_zero") {
  const double tol = 1e-12;
  CHECK(std::abs(bessel_i(0, 0.0, tol) - 1.0) < 1e-12);
  CHECK(std::abs(bessel_i(1, 0.0, tol)) < 1e-12);
  CHECK(std::abs(bessel_i(2, 0.0, tol)) < 1e-12);
}

TEST_CASE("bessel_i_matches_reference_values") {
  struct Reference {
    size_t order;
    double x;
    double expected;
  };

  const Reference refs[] = {
      {0, 1.0, 1.2660658777520082}, {1, 1.0, 0.5651591039924850}, {2, 1.0, 0.1357476697670383},
      {0, 2.0, 2.2795853023360673}, {1, 2.0, 1.5906368546373291}, {2, 2.0, 0.6889484476987382},
  };

  const double tol = 1e-9;
  for (const auto& ref : refs) {
    const double value = bessel_i(ref.order, ref.x, 1e-12);
    CHECK(std::abs(value - ref.expected) < tol);
  }
}

#if MANYBODY_HAS_STD_BESSEL_I
TEST_CASE("bessel_i_matches_std_impl") {
  const double tol = 1e-12;
  for (size_t order = 0; order < 5; ++order) {
    const double x = 0.75 * static_cast<double>(order + 1);
    const double expected = std::cyl_bessel_i(static_cast<double>(order), x);
    const double value = bessel_i(order, x, tol);
    CHECK(std::abs(value - expected) < 1e-10);
  }
}
#endif
