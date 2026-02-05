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
  using Base::Base;

  std::string to_string() const;
};

MajoranaExpression canonicalize(const MajoranaMonomial::complex_type& c,
                                const MajoranaMonomial::container_type& ops);

MajoranaExpression canonicalize(const MajoranaMonomial& term);

MajoranaExpression canonicalize(const MajoranaExpression& expr);

}  // namespace majorana
