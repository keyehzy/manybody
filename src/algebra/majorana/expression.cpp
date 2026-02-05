#include "algebra/majorana/expression.h"

#include <sstream>
#include <utility>

#include "utils/tolerances.h"

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

constexpr auto tolerance = tolerances::tolerance<MajoranaExpression::complex_type::value_type>();

MajoranaExpression canonicalize(const MajoranaMonomial::complex_type& c,
                                const MajoranaMonomial::container_type& ops) {
  if (std::norm(c) < tolerance * tolerance) {
    return {};
  }
  MajoranaProduct prod = canonicalize(ops);
  MajoranaMonomial mon(c * MajoranaExpression::complex_type(prod.sign), prod.string);
  return MajoranaExpression{mon};
}

MajoranaExpression canonicalize(const MajoranaMonomial& term) {
  return canonicalize(term.c, term.operators);
}

MajoranaExpression canonicalize(const MajoranaExpression& expr) {
  MajoranaExpression result;
  for (const auto& [ops, c] : expr.terms()) {
    result += canonicalize(c, ops);
  }
  return result;
}
}  // namespace majorana
