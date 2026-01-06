#include "algorithm/diagonal_terms.h"

#include <array>
#include <unordered_set>

namespace {

constexpr size_t kFlavorStride = Operator::kValueMask + 1;
constexpr size_t kFlavorCount = kFlavorStride * 2;

std::array<int, kFlavorCount> balance_flavors(const Term::container_type& ops) {
  std::array<int, kFlavorCount> balance{};
  for (const auto& op : ops) {
    const size_t idx = static_cast<size_t>(op.data & ~Operator::kTypeBit);
    if (op.type() == Operator::Type::Creation) {
      ++balance[idx];
    } else {
      --balance[idx];
    }
  }
  return balance;
}

std::vector<Term::container_type> diagonal_parent_candidates(const Term& term) {
  const auto balance = balance_flavors(term.operators);
  size_t pos = kFlavorCount;
  size_t neg = kFlavorCount;
  for (size_t i = 0; i < balance.size(); ++i) {
    if (balance[i] == 0) {
      continue;
    }
    if (balance[i] == 1 && pos == kFlavorCount) {
      pos = i;
      continue;
    }
    if (balance[i] == -1 && neg == kFlavorCount) {
      neg = i;
      continue;
    }
    return {};
  }

  if (pos == kFlavorCount || neg == kFlavorCount) {
    return {};
  }

  std::unordered_set<Term::container_type> seen;
  std::vector<Term::container_type> parents;
  for (size_t i = 0; i < term.operators.size(); ++i) {
    const auto& op = term.operators[i];
    const auto flavor = static_cast<Operator::ubyte>(op.data & ~Operator::kTypeBit);
    const auto type_bit = static_cast<Operator::ubyte>(op.data & Operator::kTypeBit);
    if (op.type() == Operator::Type::Creation && flavor == pos) {
      auto ops = term.operators;
      ops[i] = Operator(static_cast<Operator::ubyte>(type_bit | static_cast<Operator::ubyte>(neg)));
      Term candidate(ops);
      if (candidate.is_diagonal() && seen.insert(ops).second) {
        parents.push_back(ops);
      }
    }
    if (op.type() == Operator::Type::Annihilation && flavor == neg) {
      auto ops = term.operators;
      ops[i] = Operator(static_cast<Operator::ubyte>(type_bit | static_cast<Operator::ubyte>(pos)));
      Term candidate(ops);
      if (candidate.is_diagonal() && seen.insert(ops).second) {
        parents.push_back(ops);
      }
    }
  }
  return parents;
}

}  // namespace

DiagonalChildrenResult group_diagonal_children(const Expression& expr) {
  DiagonalChildrenResult result;
  std::unordered_set<Term::container_type> diagonal_ops;
  std::vector<Term> off_diagonals;
  for (const auto& [ops, coeff] : expr.hashmap) {
    Term term(coeff, ops);
    if (term.is_diagonal()) {
      result.diagonals.push_back(term);
      diagonal_ops.insert(ops);
      result.children.emplace(ops, std::vector<Term>{});
    } else {
      off_diagonals.push_back(term);
    }
  }

  for (const auto& term : off_diagonals) {
    const auto parents = diagonal_parent_candidates(term);
    for (const auto& parent_ops : parents) {
      if (diagonal_ops.find(parent_ops) != diagonal_ops.end()) {
        result.children[parent_ops].push_back(term);
      }
    }
  }
  return result;
}
