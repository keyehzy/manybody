#pragma once

#include <armadillo>
#include <cstddef>
#include <utility>

#include "algebra/basis.h"
#include "algebra/expression.h"

std::pair<size_t, double> cluster_by_largest_gap(const arma::vec& vals);
Expression schriffer_wolff(const Expression& kinetic, const Expression& interaction,
                           const Basis& basis, size_t iter = 1000);
