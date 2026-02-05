#pragma once

#include "algebra/expression.h"

Expression normal_order(const FermionMonomial::complex_type& c,
                        const FermionMonomial::container_type& ops);
Expression normal_order(const FermionMonomial& term);
Expression normal_order(const Expression& expr);
