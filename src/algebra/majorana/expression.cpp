#include "algebra/majorana/expression.h"

#include <sstream>
#include <utility>

namespace majorana {

MajoranaExpression::MajoranaExpression(complex_type c) {
  if (!ExpressionMap<MajoranaString>::is_zero(c)) {
    map.data.emplace(MajoranaString{}, c);
  }
}

MajoranaExpression::MajoranaExpression(int sign, const MajoranaString& str) {
  auto canonical = canonicalize(str);
  auto coeff = complex_type{static_cast<double>(sign * canonical.sign), 0.0};
  if (!ExpressionMap<MajoranaString>::is_zero(coeff)) {
    map.data.emplace(std::move(canonical.string), coeff);
  }
}

MajoranaExpression::MajoranaExpression(complex_type c, const MajoranaString& str) {
  if (ExpressionMap<MajoranaString>::is_zero(c)) {
    return;
  }
  auto canonical = canonicalize(str);
  auto coeff = c * static_cast<double>(canonical.sign);
  if (!ExpressionMap<MajoranaString>::is_zero(coeff)) {
    map.data.emplace(std::move(canonical.string), coeff);
  }
}

double MajoranaExpression::norm_squared() const {
  double result = 0.0;
  for (const auto& [str, coeff] : map.data) {
    result += std::norm(coeff);
  }
  return result;
}

MajoranaExpression& MajoranaExpression::truncate_by_norm(double min_norm) {
  map.truncate_by_norm(min_norm);
  return *this;
}

void MajoranaExpression::to_string(std::ostringstream& oss) const {
  map.format_sorted(oss, [](std::ostringstream& os, const MajoranaString& string_data,
                            const complex_type& coeff) {
    os << coeff;
    if (!string_data.empty()) {
      os << " ";
      ::majorana::to_string(os, string_data);
    }
  });
}

MajoranaExpression& MajoranaExpression::operator*=(const MajoranaExpression& value) {
  if (map.empty() || value.map.empty()) {
    map.clear();
    return *this;
  }
  map_type result;
  result.reserve(map.size() * value.map.size());
  for (const auto& [lhs_str, lhs_coeff] : map.data) {
    for (const auto& [rhs_str, rhs_coeff] : value.map.data) {
      auto product = multiply_strings(lhs_str, rhs_str);
      auto coeff = lhs_coeff * rhs_coeff * static_cast<double>(product.sign);
      ExpressionMap<MajoranaString> tmp;
      tmp.data = std::move(result);
      tmp.add(std::move(product.string), coeff);
      result = std::move(tmp.data);
    }
  }
  map.data = std::move(result);
  return *this;
}

}  // namespace majorana
