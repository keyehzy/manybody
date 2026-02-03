#pragma once

#include "algebra/majorana/majorana_expression.h"

namespace majorana {

MajoranaExpression commutator(const MajoranaExpression& A, const MajoranaExpression& B);
MajoranaExpression anticommutator(const MajoranaExpression& A, const MajoranaExpression& B);

}  // namespace majorana
