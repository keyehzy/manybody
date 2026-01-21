#include "algebra/expression.h"

#include <algorithm>
#include <sstream>
#include <utility>

#include "utils/tolerances.h"

constexpr auto tolerance = tolerances::tolerance<Expression::complex_type::value_type>();

bool Expression::is_zero(const complex_type& value) {
  return std::norm(value) < tolerance * tolerance;
}

void Expression::add_to_map(map_type& target, const container_type& ops,
                            const complex_type& coeff) {
  if (is_zero(coeff)) {
    return;
  }
  if (ops.size() > 12) {
    return;
  }
  // Use try_emplace to hash only once (find + insert in single operation)
  auto [it, inserted] = target.try_emplace(ops, coeff);
  if (!inserted) {
    it->second += coeff;
    if (is_zero(it->second)) {
      target.erase(it);
    }
  }
}

void Expression::add_to_map(map_type& target, container_type&& ops, const complex_type& coeff) {
  if (is_zero(coeff)) {
    return;
  }
  if (ops.size() > 12) {
    return;
  }
  // Use try_emplace to hash only once (find + insert in single operation)
  auto [it, inserted] = target.try_emplace(std::move(ops), coeff);
  if (!inserted) {
    it->second += coeff;
    if (is_zero(it->second)) {
      target.erase(it);
    }
  }
}

Expression::Expression(complex_type c) {
  if (!is_zero(c)) {
    hashmap.emplace(container_type{}, c);
  }
}

Expression::Expression(Operator op) {
  container_type ops{op};
  hashmap.emplace(std::move(ops), complex_type{1.0, 0.0});
}

Expression::Expression(const Term& term) {
  if (!is_zero(term.c)) {
    hashmap.emplace(term.operators, term.c);
  }
}

Expression::Expression(Term&& term) {
  if (!is_zero(term.c)) {
    hashmap.emplace(std::move(term.operators), term.c);
  }
}

Expression::Expression(const container_type& container) {
  hashmap.emplace(container, complex_type{1.0, 0.0});
}

Expression::Expression(container_type&& container) {
  hashmap.emplace(std::move(container), complex_type{1.0, 0.0});
}

Expression::Expression(std::initializer_list<Term> lst) {
  for (const auto& term : lst) {
    add_to_map(hashmap, term.operators, term.c);
  }
}

Expression Expression::adjoint() const {
  Expression result;
  for (const auto& [ops, coeff] : hashmap) {
    Term term(coeff, ops);
    Term adj = term.adjoint();
    add_to_map(result.hashmap, std::move(adj.operators), adj.c);
  }
  return result;
}

size_t Expression::size() const { return hashmap.size(); }

Expression& Expression::truncate_by_size(size_t max_size) {
  if (max_size == 0) {
    hashmap.clear();
    return *this;
  }
  for (auto it = hashmap.begin(); it != hashmap.end();) {
    if (it->first.size() > max_size) {
      it = hashmap.erase(it);
    } else {
      ++it;
    }
  }
  return *this;
}

Expression& Expression::truncate_by_norm(double min_norm) {
  if (min_norm <= 0.0) {
    return *this;
  }
  const auto cutoff = static_cast<complex_type::value_type>(min_norm);
  const auto cutoff_norm = cutoff * cutoff;
  for (auto it = hashmap.begin(); it != hashmap.end();) {
    if (std::norm(it->second) < cutoff_norm) {
      it = hashmap.erase(it);
    } else {
      ++it;
    }
  }
  return *this;
}

Expression& Expression::filter_by_size(size_t size) {
  if (size == 0) {
    hashmap.clear();
    return *this;
  }
  for (auto it = hashmap.begin(); it != hashmap.end();) {
    if (it->first.size() != size) {
      it = hashmap.erase(it);
    } else {
      ++it;
    }
  }
  return *this;
}

void Expression::to_string(std::ostringstream& oss) const {
  if (hashmap.empty()) {
    oss << "0";
    return;
  }
  std::vector<const map_type::value_type*> ordered;
  ordered.reserve(hashmap.size());
  for (const auto& entry : hashmap) {
    ordered.push_back(&entry);
  }
  std::sort(ordered.begin(), ordered.end(),
            [](const map_type::value_type* left, const map_type::value_type* right) {
              const auto left_size = left->first.size();
              const auto right_size = right->first.size();
              if (left_size != right_size) {
                return left_size < right_size;
              }
              const auto left_norm = std::norm(left->second);
              const auto right_norm = std::norm(right->second);
              return left_norm > right_norm;
            });

  bool first = true;
  for (const auto* entry : ordered) {
    Term term(entry->second, entry->first);
    if (!first) {
      oss << "\n";
    }
    term.to_string(oss);
    first = false;
  }
}

std::string Expression::to_string() const {
  std::ostringstream oss;
  to_string(oss);
  return oss.str();
}

Expression& Expression::operator+=(const complex_type& value) {
  add_to_map(hashmap, container_type{}, value);
  return *this;
}

Expression& Expression::operator-=(const complex_type& value) {
  add_to_map(hashmap, container_type{}, -value);
  return *this;
}

Expression& Expression::operator*=(const complex_type& value) {
  if (is_zero(value)) {
    hashmap.clear();
    return *this;
  }
  for (auto& [ops, coeff] : hashmap) {
    coeff *= value;
  }
  return *this;
}

Expression& Expression::operator/=(const complex_type& value) {
  for (auto& [ops, coeff] : hashmap) {
    coeff /= value;
  }
  return *this;
}

Expression& Expression::operator+=(const Expression& value) {
  for (const auto& [ops, coeff] : value.hashmap) {
    add_to_map(hashmap, ops, coeff);
  }
  return *this;
}

Expression& Expression::operator-=(const Expression& value) {
  for (const auto& [ops, coeff] : value.hashmap) {
    add_to_map(hashmap, ops, -coeff);
  }
  return *this;
}

Expression& Expression::operator*=(const Expression& value) {
  if (hashmap.empty() || value.hashmap.empty()) {
    hashmap.clear();
    return *this;
  }
  map_type result;
  result.reserve(hashmap.size() * value.hashmap.size());
  for (const auto& [lhs_ops, lhs_coeff] : hashmap) {
    for (const auto& [rhs_ops, rhs_coeff] : value.hashmap) {
      if (lhs_ops.size() + rhs_ops.size() > 12) {
        continue;
      }
      container_type combined = lhs_ops;
      combined.append_range(rhs_ops.begin(), rhs_ops.end());
      add_to_map(result, std::move(combined), lhs_coeff * rhs_coeff);
    }
  }
  hashmap = std::move(result);
  return *this;
}

Expression& Expression::operator+=(const Term& value) {
  add_to_map(hashmap, value.operators, value.c);
  return *this;
}

Expression& Expression::operator-=(const Term& value) {
  add_to_map(hashmap, value.operators, -value.c);
  return *this;
}

Expression& Expression::operator*=(const Term& value) {
  if (hashmap.empty()) {
    return *this;
  }
  if (is_zero(value.c)) {
    hashmap.clear();
    return *this;
  }

  if (value.operators.size() == 0) {
    for (auto& [ops, coeff] : hashmap) {
      coeff *= value.c;
    }
    return *this;
  }
  map_type result;
  result.reserve(hashmap.size());
  for (const auto& [ops, coeff] : hashmap) {
    if (ops.size() + value.operators.size() > 12) {
      continue;
    }
    container_type combined = ops;
    combined.append_range(value.operators.begin(), value.operators.end());
    add_to_map(result, std::move(combined), coeff * value.c);
  }
  hashmap = std::move(result);
  return *this;
}
