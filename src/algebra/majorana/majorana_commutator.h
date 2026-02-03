#pragma once

#include "algebra/majorana/majorana_expression.h"

MajoranaExpression commutator(const MajoranaExpression& A, const MajoranaExpression& B);
MajoranaExpression anticommutator(const MajoranaExpression& A, const MajoranaExpression& B);
