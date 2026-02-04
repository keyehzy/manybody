#include "algorithms/diagonal_terms.h"

#include <algorithm>

namespace {

bool has_matching_substring(const FermionString& diagonal_ops,
                            const FermionString& off_diagonal_ops) {
  if (diagonal_ops.empty() || off_diagonal_ops.empty()) {
    return false;
  }
  const size_t required_len = std::max<size_t>(1, diagonal_ops.size() / 2);
  if (off_diagonal_ops.size() < required_len) {
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

bool is_term_contained(const FermionString& small, const FermionString& large) {
  if (small.empty()) {
    return false;
  }
  if (small.size() > large.size()) {
    return false;
  }

  std::vector<bool> used(large.size(), false);

  for (const auto& op : small) {
    bool found = false;
    for (size_t i = 0; i < large.size(); ++i) {
      if (!used[i] && op == large[i]) {
        used[i] = true;
        found = true;
        break;
      }
    }
    if (!found) {
      return false;
    }
  }
  return true;
}

}  // namespace

std::vector<Term> find_matching_terms(const Term& term, const Expression& expr) {
  std::vector<Term> matches;
  for (const auto& [ops, coeff] : expr.terms()) {
    Term candidate(coeff, ops);
    if (is_term_contained(term.operators, candidate.operators)) {
      matches.push_back(candidate);
    }
  }
  return matches;
}

DiagonalChildrenResult group_diagonal_children(const Expression& expr) {
  DiagonalChildrenResult result;
  std::vector<Term> off_diagonals;
  for (const auto& [ops, coeff] : expr.terms()) {
    Term term(coeff, ops);
    if (is_diagonal(term)) {
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
