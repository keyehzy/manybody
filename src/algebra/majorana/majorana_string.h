#pragma once

#include <cstddef>
#include <cstdint>
#include <sstream>
#include <string>

#include "algebra/majorana/majorana_operator.h"
#include "utils/static_vector.h"

using MajoranaString = static_vector<MajoranaOperator, 24, std::uint8_t>;

namespace majorana_string {
void to_string(std::ostringstream& oss, const MajoranaString& str);

std::string to_string(const MajoranaString& str);
}  // namespace majorana_string

struct MajoranaProduct {
  int sign = 1;
  MajoranaString string{};
};

/// Multiply two sorted Majorana strings using the Clifford algebra relation
/// {gamma_i, gamma_j} = 2 * delta_ij.  Each pair of equal indices cancels
/// (gamma_i^2 = 1) and every anticommutation swap contributes a sign flip.
MajoranaProduct multiply_strings(const MajoranaString& a, const MajoranaString& b) noexcept;

namespace majorana_string {
MajoranaProduct canonicalize(const MajoranaString& str) noexcept;
}  // namespace majorana_string
