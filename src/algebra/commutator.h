#pragma once

#include "algebra/expression.h"

Expression commutator(const FermionMonomial& A, const FermionMonomial& B);
Expression commutator(const Expression& A, const Expression& B);
Expression anticommutator(const FermionMonomial& A, const FermionMonomial& B);
Expression anticommutator(const Expression& A, const Expression& B);
Expression BCH(const Expression& A, const Expression& B,
               Expression::complex_type::value_type lambda, size_t order = 1000);
