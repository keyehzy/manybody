#pragma once

#include <complex>
#include <cstddef>
#include <cstdint>
#include <sstream>
#include <string>

#include "algebra/majorana/operator.h"
#include "algebra/monomial.h"
#include "utils/static_vector.h"

namespace majorana {

struct MajoranaMonomial
    : MonomialBase<MajoranaMonomial, MajoranaOperator, 24, std::uint8_t, std::complex<double>> {
  using complex_type = std::complex<double>;
  using container_type = MonomialBase::container_type;

  constexpr MajoranaMonomial() noexcept = default;
  constexpr ~MajoranaMonomial() noexcept = default;

  constexpr MajoranaMonomial(const MajoranaMonomial& other) noexcept = default;
  constexpr MajoranaMonomial& operator=(const MajoranaMonomial& other) noexcept = default;
  constexpr MajoranaMonomial(MajoranaMonomial&& other) noexcept = default;
  constexpr MajoranaMonomial& operator=(MajoranaMonomial&& other) noexcept = default;

  using MonomialBase::MonomialBase;
};

void to_string(std::ostringstream& oss, const MajoranaMonomial::container_type& str);

std::string to_string(const MajoranaMonomial::container_type& str);

struct MajoranaProduct {
  int sign = 1;
  MajoranaMonomial::container_type string{};
};

/// Multiply two sorted Majorana strings using the Clifford algebra relation
/// {gamma_i, gamma_j} = 2 * delta_ij.  Each pair of equal indices cancels
/// (gamma_i^2 = 1) and every anticommutation swap contributes a sign flip.
MajoranaProduct multiply_strings(const MajoranaMonomial::container_type& a,
                                 const MajoranaMonomial::container_type& b) noexcept;

MajoranaProduct canonicalize(const MajoranaMonomial::container_type& str) noexcept;

}  // namespace majorana
