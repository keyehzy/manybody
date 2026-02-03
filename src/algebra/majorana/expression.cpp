#include "algebra/majorana/expression.h"

#include <algorithm>
#include <sstream>
#include <utility>
#include <vector>

#include "utils/tolerances.h"

namespace majorana {

constexpr auto tolerance = tolerances::tolerance<MajoranaExpression::complex_type::value_type>();

bool MajoranaExpression::is_zero(const complex_type& value) {
  return std::norm(value) < tolerance * tolerance;
}

void MajoranaExpression::add_to_map(map_type& target, const MajoranaString& str,
                                    const complex_type& coeff) {
  if (is_zero(coeff)) {
    return;
  }
  auto [it, inserted] = target.try_emplace(str, coeff);
  if (!inserted) {
    it->second += coeff;
    if (is_zero(it->second)) {
      target.erase(it);
    }
  }
}

void MajoranaExpression::add_to_map(map_type& target, MajoranaString&& str,
                                    const complex_type& coeff) {
  if (is_zero(coeff)) {
    return;
  }
  auto [it, inserted] = target.try_emplace(std::move(str), coeff);
  if (!inserted) {
    it->second += coeff;
    if (is_zero(it->second)) {
      target.erase(it);
    }
  }
}

MajoranaExpression::MajoranaExpression(complex_type c) {
  if (!is_zero(c)) {
    hashmap.emplace(MajoranaString{}, c);
  }
}

MajoranaExpression::MajoranaExpression(int sign, const MajoranaString& str) {
  auto canonical = canonicalize(str);
  auto coeff = complex_type{static_cast<double>(sign * canonical.sign), 0.0};
  if (!is_zero(coeff)) {
    hashmap.emplace(std::move(canonical.string), coeff);
  }
}

MajoranaExpression::MajoranaExpression(complex_type c, const MajoranaString& str) {
  if (is_zero(c)) {
    return;
  }
  auto canonical = canonicalize(str);
  auto coeff = c * static_cast<double>(canonical.sign);
  if (!is_zero(coeff)) {
    hashmap.emplace(std::move(canonical.string), coeff);
  }
}

size_t MajoranaExpression::size() const { return hashmap.size(); }

double MajoranaExpression::norm_squared() const {
  double result = 0.0;
  for (const auto& [str, coeff] : hashmap) {
    result += std::norm(coeff);
  }
  return result;
}

MajoranaExpression& MajoranaExpression::truncate_by_norm(double min_norm) {
  if (min_norm <= 0.0) {
    return *this;
  }
  const auto cutoff_norm = min_norm * min_norm;
  for (auto it = hashmap.begin(); it != hashmap.end();) {
    if (std::norm(it->second) < cutoff_norm) {
      it = hashmap.erase(it);
    } else {
      ++it;
    }
  }
  return *this;
}

void MajoranaExpression::to_string(std::ostringstream& oss) const {
  if (hashmap.empty()) {
    oss << "0";
    return;
  }
  std::vector<const map_type::value_type*> ordered;
  ordered.reserve(hashmap.size());
  for (const auto& entry : hashmap) {
    ordered.push_back(&entry);
  }
  std::sort(ordered.begin(), ordered.end(),
            [](const map_type::value_type* left, const map_type::value_type* right) {
              const auto left_size = left->first.size();
              const auto right_size = right->first.size();
              if (left_size != right_size) {
                return left_size < right_size;
              }
              return std::norm(left->second) > std::norm(right->second);
            });

  bool first = true;
  for (const auto* entry : ordered) {
    if (!first) {
      oss << "\n";
    }
    const auto& coeff = entry->second;
    oss << coeff;
    const auto& string_data = entry->first;
    if (!string_data.empty()) {
      oss << " ";
      ::majorana::to_string(oss, string_data);
    }
    first = false;
  }
}

std::string MajoranaExpression::to_string() const {
  std::ostringstream oss;
  to_string(oss);
  return oss.str();
}

MajoranaExpression& MajoranaExpression::operator+=(const complex_type& value) {
  add_to_map(hashmap, MajoranaString{}, value);
  return *this;
}

MajoranaExpression& MajoranaExpression::operator-=(const complex_type& value) {
  add_to_map(hashmap, MajoranaString{}, -value);
  return *this;
}

MajoranaExpression& MajoranaExpression::operator*=(const complex_type& value) {
  if (is_zero(value)) {
    hashmap.clear();
    return *this;
  }
  for (auto& [str, coeff] : hashmap) {
    coeff *= value;
  }
  return *this;
}

MajoranaExpression& MajoranaExpression::operator/=(const complex_type& value) {
  for (auto& [str, coeff] : hashmap) {
    coeff /= value;
  }
  return *this;
}

MajoranaExpression& MajoranaExpression::operator+=(const MajoranaExpression& value) {
  for (const auto& [str, coeff] : value.hashmap) {
    add_to_map(hashmap, str, coeff);
  }
  return *this;
}

MajoranaExpression& MajoranaExpression::operator-=(const MajoranaExpression& value) {
  for (const auto& [str, coeff] : value.hashmap) {
    add_to_map(hashmap, str, -coeff);
  }
  return *this;
}

MajoranaExpression& MajoranaExpression::operator*=(const MajoranaExpression& value) {
  if (hashmap.empty() || value.hashmap.empty()) {
    hashmap.clear();
    return *this;
  }
  map_type result;
  result.reserve(hashmap.size() * value.hashmap.size());
  for (const auto& [lhs_str, lhs_coeff] : hashmap) {
    for (const auto& [rhs_str, rhs_coeff] : value.hashmap) {
      auto product = multiply_strings(lhs_str, rhs_str);
      auto coeff = lhs_coeff * rhs_coeff * static_cast<double>(product.sign);
      add_to_map(result, std::move(product.string), coeff);
    }
  }
  hashmap = std::move(result);
  return *this;
}

}  // namespace majorana
