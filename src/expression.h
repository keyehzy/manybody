#pragma once

#include <algorithm>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "term.h"

struct Expression {
  using complex_type = Term::complex_type;
  using container_type = Term::container_type;
  using map_type = std::unordered_map<container_type, complex_type>;

  map_type hashmap;

  Expression() = default;
  ~Expression() = default;

  Expression(const Expression&) = default;
  Expression& operator=(const Expression&) = default;
  Expression(Expression&&) noexcept = default;
  Expression& operator=(Expression&&) noexcept = default;

  explicit Expression(complex_type c);
  explicit Expression(Operator op);
  explicit Expression(const Term& term);
  explicit Expression(Term&& term);
  explicit Expression(const container_type& container);
  explicit Expression(container_type&& container);
  explicit Expression(std::initializer_list<Term> lst);

  template <typename Container>
  Expression(complex_type c, Container&& ops) {
    hashmap.emplace(std::forward<Container>(ops), c);
  }

  Expression adjoint() const;

  size_t size() const { return hashmap.size(); }

  std::string to_string() const;

  void normalize();

  Expression& operator+=(const complex_type& value);
  Expression& operator-=(const complex_type& value);
  Expression& operator*=(const complex_type& value);
  Expression& operator/=(const complex_type& value);

  Expression& operator+=(const Expression& value);
  Expression& operator-=(const Expression& value);
  Expression& operator*=(const Expression& value);

  Expression& operator+=(const Term& value);
  Expression& operator-=(const Term& value);
  Expression& operator*=(const Term& value);

 private:
  static bool is_zero(const complex_type& value) {
    return value == complex_type{};
  }

  static bool less_ops(const container_type& left,
                       const container_type& right) {
    if (left.size() != right.size()) {
      return left.size() < right.size();
    }
    for (size_t i = 0; i < left.size(); ++i) {
      if (left[i] < right[i]) {
        return true;
      }
      if (right[i] < left[i]) {
        return false;
      }
    }
    return false;
  }

  static void add_to_map(map_type& target, const container_type& ops,
                         const complex_type& coeff) {
    if (is_zero(coeff)) {
      return;
    }
    auto it = target.find(ops);
    if (it == target.end()) {
      target.emplace(ops, coeff);
      return;
    }
    it->second += coeff;
    if (is_zero(it->second)) {
      target.erase(it);
    }
  }

  static void add_to_map(map_type& target, container_type&& ops,
                         const complex_type& coeff) {
    if (is_zero(coeff)) {
      return;
    }
    auto it = target.find(ops);
    if (it == target.end()) {
      target.emplace(std::move(ops), coeff);
      return;
    }
    it->second += coeff;
    if (is_zero(it->second)) {
      target.erase(it);
    }
  }
};

inline Expression::Expression(complex_type c) {
  if (!is_zero(c)) {
    hashmap.emplace(container_type{}, c);
  }
}

inline Expression::Expression(Operator op) {
  container_type ops{op};
  hashmap.emplace(std::move(ops), complex_type{1.0f, 0.0f});
}

inline Expression::Expression(const Term& term) {
  if (!is_zero(term.c)) {
    hashmap.emplace(term.operators, term.c);
  }
}

inline Expression::Expression(Term&& term) {
  if (!is_zero(term.c)) {
    hashmap.emplace(std::move(term.operators), term.c);
  }
}

inline Expression::Expression(const container_type& container) {
  hashmap.emplace(container, complex_type{1.0f, 0.0f});
}

inline Expression::Expression(container_type&& container) {
  hashmap.emplace(std::move(container), complex_type{1.0f, 0.0f});
}

inline Expression::Expression(std::initializer_list<Term> lst) {
  for (const auto& term : lst) {
    add_to_map(hashmap, term.operators, term.c);
  }
}

inline Expression Expression::adjoint() const {
  Expression result;
  for (const auto& [ops, coeff] : hashmap) {
    Term term(coeff, ops);
    Term adj = term.adjoint();
    add_to_map(result.hashmap, std::move(adj.operators), adj.c);
  }
  return result;
}

inline Expression& Expression::operator+=(const complex_type& value) {
  add_to_map(hashmap, container_type{}, value);
  return *this;
}

inline Expression& Expression::operator-=(const complex_type& value) {
  add_to_map(hashmap, container_type{}, -value);
  return *this;
}

inline Expression& Expression::operator*=(const complex_type& value) {
  if (is_zero(value)) {
    hashmap.clear();
    return *this;
  }
  for (auto& [ops, coeff] : hashmap) {
    coeff *= value;
  }
  return *this;
}

inline Expression& Expression::operator/=(const complex_type& value) {
  for (auto& [ops, coeff] : hashmap) {
    coeff /= value;
  }
  return *this;
}

inline Expression& Expression::operator+=(const Expression& value) {
  for (const auto& [ops, coeff] : value.hashmap) {
    add_to_map(hashmap, ops, coeff);
  }
  return *this;
}

inline Expression& Expression::operator-=(const Expression& value) {
  for (const auto& [ops, coeff] : value.hashmap) {
    add_to_map(hashmap, ops, -coeff);
  }
  return *this;
}

inline Expression& Expression::operator*=(const Expression& value) {
  if (hashmap.empty() || value.hashmap.empty()) {
    hashmap.clear();
    return *this;
  }
  map_type result;
  result.reserve(hashmap.size() * value.hashmap.size());
  for (const auto& [lhs_ops, lhs_coeff] : hashmap) {
    for (const auto& [rhs_ops, rhs_coeff] : value.hashmap) {
      container_type combined = lhs_ops;
      combined.append_range(rhs_ops.begin(), rhs_ops.end());
      add_to_map(result, std::move(combined), lhs_coeff * rhs_coeff);
    }
  }
  hashmap = std::move(result);
  return *this;
}

inline Expression& Expression::operator+=(const Term& value) {
  add_to_map(hashmap, value.operators, value.c);
  return *this;
}

inline Expression& Expression::operator-=(const Term& value) {
  add_to_map(hashmap, value.operators, -value.c);
  return *this;
}

inline Expression& Expression::operator*=(const Term& value) {
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
    container_type combined = ops;
    combined.append_range(value.operators.begin(), value.operators.end());
    add_to_map(result, std::move(combined), coeff * value.c);
  }
  hashmap = std::move(result);
  return *this;
}
