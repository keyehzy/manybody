#include "algorithm/diagonal_terms.h"

namespace {

bool has_matching_substring(const Term::container_type& diagonal_ops,
                            const Term::container_type& off_diagonal_ops) {
  if (diagonal_ops.empty() || off_diagonal_ops.empty()) {
    return false;
  }
  const size_t required_len = diagonal_ops.size() / 2;
  if (required_len == 0 || off_diagonal_ops.size() < required_len) {
    return false;
  }

  for (size_t i = 0; i + required_len <= diagonal_ops.size(); ++i) {
    for (size_t j = 0; j + required_len <= off_diagonal_ops.size(); ++j) {
      bool match = true;
      for (size_t k = 0; k < required_len; ++k) {
        if (!(diagonal_ops[i + k] == off_diagonal_ops[j + k])) {
          match = false;
          break;
        }
      }
      if (match) {
        return true;
      }
    }
  }
  return false;
}

}  // namespace

DiagonalChildrenResult group_diagonal_children(const Expression& expr) {
  DiagonalChildrenResult result;
  std::vector<Term> off_diagonals;
  for (const auto& [ops, coeff] : expr.hashmap) {
    Term term(coeff, ops);
    if (term.is_diagonal()) {
      result.diagonals.push_back(term);
      result.children.emplace(ops, std::vector<Term>{});
    } else {
      off_diagonals.push_back(term);
    }
  }

  for (const auto& diagonal : result.diagonals) {
    for (const auto& off_diagonal : off_diagonals) {
      if (has_matching_substring(diagonal.operators, off_diagonal.operators)) {
        result.children[diagonal.operators].push_back(off_diagonal);
      }
    }
  }
  return result;
}
