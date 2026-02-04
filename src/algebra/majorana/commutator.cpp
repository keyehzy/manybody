#include "algebra/majorana/commutator.h"

namespace majorana {

MajoranaExpression commutator(const MajoranaExpression& A, const MajoranaExpression& B) {
  // [A, B] = AB - BA
  // For each pair of Majorana strings (a, b):
  //   ab and ba produce the same merged string but potentially different signs.
  //   If sign(ab) == sign(ba), the pair cancels in the commutator.
  //   If sign(ab) != sign(ba), the pair contributes 2 * sign(ab) * ca * cb * string.
  MajoranaExpression result;
  for (const auto& [str_a, coeff_a] : A.map.data) {
    for (const auto& [str_b, coeff_b] : B.map.data) {
      auto ab = multiply_strings(str_a, str_b);
      auto ba = multiply_strings(str_b, str_a);

      if (ab.sign != ba.sign) {
        auto coeff = 2.0 * static_cast<double>(ab.sign) * coeff_a * coeff_b;
        result.map.add(std::move(ab.string), coeff);
      }
    }
  }
  return result;
}

MajoranaExpression anticommutator(const MajoranaExpression& A, const MajoranaExpression& B) {
  // {A, B} = AB + BA
  // For each pair of Majorana strings (a, b):
  //   If sign(ab) == sign(ba), contributes 2 * sign(ab) * ca * cb * string.
  //   If sign(ab) != sign(ba), the pair cancels in the anticommutator.
  MajoranaExpression result;
  for (const auto& [str_a, coeff_a] : A.map.data) {
    for (const auto& [str_b, coeff_b] : B.map.data) {
      auto ab = multiply_strings(str_a, str_b);
      auto ba = multiply_strings(str_b, str_a);

      if (ab.sign == ba.sign) {
        auto coeff = 2.0 * static_cast<double>(ab.sign) * coeff_a * coeff_b;
        result.map.add(std::move(ab.string), coeff);
      }
    }
  }
  return result;
}

}  // namespace majorana
