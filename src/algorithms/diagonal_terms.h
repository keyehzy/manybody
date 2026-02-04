#pragma once

#include <vector>

#include "algebra/expression.h"
#include "robin_hood.h"

struct DiagonalChildrenResult {
  std::vector<FermionMonomial> diagonals{};
  robin_hood::unordered_map<FermionMonomial::container_type, std::vector<FermionMonomial>>
      children{};
};

std::vector<FermionMonomial> find_matching_terms(const FermionMonomial& term,
                                                 const Expression& expr);
DiagonalChildrenResult group_diagonal_children(const Expression& expr);
