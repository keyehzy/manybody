#pragma once

#include "algebra/fermion/expression.h"
#include "algebra/fourier_transform.h"

inline Expression fourier_transform_operator(Operator op, const Index& index,
                                             FourierMode mode = FourierMode::Direct) {
  return fourier_transform_operator<Expression>(op, index, mode);
}
