#pragma once

#include <vector>

#include "algebra/expression.h"
#include "robin_hood.h"

struct DiagonalChildrenResult {
  std::vector<Term> diagonals;
  robin_hood::unordered_map<Term::container_type, std::vector<Term>> children;
};

std::vector<Term> find_matching_terms(const Term& term, const Expression& expr);
DiagonalChildrenResult group_diagonal_children(const Expression& expr);
