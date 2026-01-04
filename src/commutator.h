#pragma once

#include "expression.h"

Expression commutator(const Term& A, const Term& B);
Expression commutator(const Expression& A, const Expression& B);
Expression anticommutator(const Term& A, const Term& B);
Expression anticommutator(const Expression& A, const Expression& B);
Expression BCH(const Expression& A, const Expression& B,
               Expression::complex_type::value_type lambda, size_t order = 1000);
