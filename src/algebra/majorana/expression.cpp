#include "algebra/majorana/expression.h"

#include <sstream>
#include <utility>

namespace majorana {

std::string MajoranaExpression::to_string() const {
  return Base::to_string(
      [](std::ostringstream& os, const container_type& string_data, const complex_type& coeff) {
        os << coeff;
        if (!string_data.empty()) {
          os << " ";
          ::majorana::to_string(os, string_data);
        }
      });
}

MajoranaExpression& MajoranaExpression::operator*=(const MajoranaExpression& value) {
  if (this->empty() || value.empty()) {
    this->clear();
    return *this;
  }
  map_type result;
  result.reserve(this->size() * value.size());
  for (const auto& [lhs_str, lhs_coeff] : this->data) {
    for (const auto& [rhs_str, rhs_coeff] : value.data) {
      auto product = multiply_strings(lhs_str, rhs_str);
      auto coeff = lhs_coeff * rhs_coeff * static_cast<double>(product.sign);
      add_to(result, std::move(product.string), coeff);
    }
  }
  this->data = std::move(result);
  return *this;
}

MajoranaExpression& MajoranaExpression::operator*=(const MajoranaMonomial& value) {
  if (this->empty()) {
    return *this;
  }
  if (this->is_zero(value.c)) {
    this->clear();
    return *this;
  }

  auto canonical = canonicalize(value.operators);
  auto coeff = value.c * static_cast<double>(canonical.sign);
  if (this->is_zero(coeff)) {
    this->clear();
    return *this;
  }
  if (canonical.string.empty()) {
    this->scale(coeff);
    return *this;
  }

  map_type result;
  result.reserve(this->size());
  for (const auto& [lhs_str, lhs_coeff] : this->data) {
    auto product = multiply_strings(lhs_str, canonical.string);
    auto scaled = lhs_coeff * coeff * static_cast<double>(product.sign);
    add_to(result, std::move(product.string), scaled);
  }
  this->data = std::move(result);
  return *this;
}

}  // namespace majorana
