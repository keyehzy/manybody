#pragma once

#include "algebra/expression.h"
#include "algebra/majorana/majorana_expression.h"

namespace majorana {

MajoranaExpression to_majorana(const Expression& expr);
Expression from_majorana(const MajoranaExpression& expr);

}  // namespace majorana
