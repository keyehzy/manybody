#pragma once

#include <complex>
#include <sstream>
#include <string>

#include "algebra/expression_base.h"
#include "algebra/majorana/string.h"
#include "robin_hood.h"

namespace majorana {

struct MajoranaExpression : ExpressionBase<MajoranaExpression, MajoranaMonomial> {
  using Base = ExpressionBase<MajoranaExpression, MajoranaMonomial>;
  using Base::operator*=;
  using Base::operator+=;
  using Base::operator-=;

  void add_to_map(const container_type& str, const complex_type& coeff) {
    if (this->is_zero(coeff)) {
      return;
    }
    auto canonical = canonicalize(str);
    auto scaled = coeff * static_cast<double>(canonical.sign);
    if (this->is_zero(scaled)) {
      return;
    }
    this->add(std::move(canonical.string), scaled);
  }

  MajoranaExpression() = default;

  explicit MajoranaExpression(complex_type c) {
    if (!is_zero(c)) {
      data.emplace(container_type{}, c);
    }
  }

  explicit MajoranaExpression(const MajoranaMonomial& term) { add_to_map(term.operators, term.c); }

  explicit MajoranaExpression(MajoranaMonomial&& term) {
    add_to_map(std::move(term.operators), term.c);
  }

  explicit MajoranaExpression(int sign, const container_type& str) {
    add_to_map(str, complex_type{static_cast<double>(sign), 0.0});
  }

  explicit MajoranaExpression(complex_type c, const container_type& str) { add_to_map(str, c); }

  MajoranaExpression& operator*=(const MajoranaExpression& value);
  MajoranaExpression& operator*=(const MajoranaMonomial& value);

  std::string to_string() const;
};

}  // namespace majorana
