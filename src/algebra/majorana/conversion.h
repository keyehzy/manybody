#pragma once

#include "algebra/fermion/expression.h"
#include "algebra/majorana/expression.h"

namespace majorana {

MajoranaExpression to_majorana(const FermionExpression& expr);
FermionExpression from_majorana(const MajoranaExpression& expr);

}  // namespace majorana
