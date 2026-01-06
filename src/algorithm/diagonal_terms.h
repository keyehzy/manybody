#pragma once

#include <unordered_map>
#include <vector>

#include "expression.h"

struct DiagonalChildrenResult {
  std::vector<Term> diagonals;
  std::unordered_map<Term::container_type, std::vector<Term>> children;
};

DiagonalChildrenResult group_diagonal_children(const Expression& expr);
