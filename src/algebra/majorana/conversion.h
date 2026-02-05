#pragma once

#include "algebra/fermion/expression.h"
#include "algebra/majorana/expression.h"

namespace majorana {

MajoranaExpression to_majorana(const Expression& expr);
Expression from_majorana(const MajoranaExpression& expr);

}  // namespace majorana
