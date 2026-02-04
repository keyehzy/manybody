#include "algebra/majorana/expression.h"

#include <sstream>
#include <utility>

namespace majorana {

void MajoranaExpression::add_to_map(ExpressionMap<MajoranaString>& target,
                                    const MajoranaString& str, const complex_type& coeff) {
  if (ExpressionMap<MajoranaString>::is_zero(coeff)) {
    return;
  }
  auto canonical = canonicalize(str);
  auto scaled = coeff * static_cast<double>(canonical.sign);
  if (ExpressionMap<MajoranaString>::is_zero(scaled)) {
    return;
  }
  target.add(std::move(canonical.string), scaled);
}

MajoranaExpression::MajoranaExpression(complex_type c) {
  if (!ExpressionMap<MajoranaString>::is_zero(c)) {
    map.data.emplace(MajoranaString{}, c);
  }
}

MajoranaExpression::MajoranaExpression(int sign, const MajoranaString& str) {
  add_to_map(map, str, complex_type{static_cast<double>(sign), 0.0});
}

MajoranaExpression::MajoranaExpression(complex_type c, const MajoranaString& str) {
  add_to_map(map, str, c);
}

MajoranaExpression::MajoranaExpression(const MajoranaTerm& term) {
  add_to_map(map, term.operators, term.c);
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
  ExpressionMap<MajoranaString> result;
  result.reserve(map.size() * value.map.size());
  for (const auto& [lhs_str, lhs_coeff] : map.data) {
    for (const auto& [rhs_str, rhs_coeff] : value.map.data) {
      auto product = multiply_strings(lhs_str, rhs_str);
      auto coeff = lhs_coeff * rhs_coeff * static_cast<double>(product.sign);
      result.add(std::move(product.string), coeff);
    }
  }
  map.data = std::move(result.data);
  return *this;
}

MajoranaExpression& MajoranaExpression::operator+=(const MajoranaTerm& value) {
  add_to_map(map, value.operators, value.c);
  return *this;
}

MajoranaExpression& MajoranaExpression::operator-=(const MajoranaTerm& value) {
  add_to_map(map, value.operators, -value.c);
  return *this;
}

MajoranaExpression& MajoranaExpression::operator*=(const MajoranaTerm& value) {
  if (map.empty()) {
    return *this;
  }
  if (ExpressionMap<MajoranaString>::is_zero(value.c)) {
    map.clear();
    return *this;
  }

  auto canonical = canonicalize(value.operators);
  auto coeff = value.c * static_cast<double>(canonical.sign);
  if (ExpressionMap<MajoranaString>::is_zero(coeff)) {
    map.clear();
    return *this;
  }
  if (canonical.string.empty()) {
    map *= coeff;
    return *this;
  }

  ExpressionMap<MajoranaString> result;
  result.reserve(map.size());
  for (const auto& [lhs_str, lhs_coeff] : map.data) {
    auto product = multiply_strings(lhs_str, canonical.string);
    auto scaled = lhs_coeff * coeff * static_cast<double>(product.sign);
    result.add(std::move(product.string), scaled);
  }
  map.data = std::move(result.data);
  return *this;
}

}  // namespace majorana
