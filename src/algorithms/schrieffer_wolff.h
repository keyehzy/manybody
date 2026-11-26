#pragma once

#include <armadillo>
#include <cstddef>
#include <utility>

#include "algebra/fermion/basis.h"
#include "algebra/fermion/expression.h"

std::pair<size_t, double> cluster_by_largest_gap(const arma::vec& vals);
FermionExpression schriffer_wolff(const FermionExpression& kinetic,
                                  const FermionExpression& interaction, const FermionBasis& basis,
                                  size_t iter = 1000);
