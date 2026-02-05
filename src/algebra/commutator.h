#pragma once

#include "algebra/expression.h"

Expression BCH(const Expression& A, const Expression& B,
               Expression::complex_type::value_type lambda, size_t order = 1000);
