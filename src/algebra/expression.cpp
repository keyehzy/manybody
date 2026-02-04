#include "algebra/expression.h"

#include <algorithm>
#include <sstream>
#include <utility>

void Expression::add_to_map(ExpressionMap<container_type>& target, const container_type& ops,
                            const complex_type& coeff) {
  if (ops.size() > Term::static_vector_size) {
    return;
  }
  target.add(ops, coeff);
}

void Expression::add_to_map(ExpressionMap<container_type>& target, container_type&& ops,
                            const complex_type& coeff) {
  if (ops.size() > Term::static_vector_size) {
    return;
  }
  target.add(std::move(ops), coeff);
}

Expression::Expression(complex_type c) {
  if (!ExpressionMap<container_type>::is_zero(c)) {
    map.data.emplace(container_type{}, c);
  }
}

Expression::Expression(Operator op) {
  container_type ops{op};
  map.data.emplace(std::move(ops), complex_type{1.0, 0.0});
}

Expression::Expression(const Term& term) {
  if (!ExpressionMap<container_type>::is_zero(term.c)) {
    map.data.emplace(term.operators, term.c);
  }
}

Expression::Expression(Term&& term) {
  if (!ExpressionMap<container_type>::is_zero(term.c)) {
    map.data.emplace(std::move(term.operators), term.c);
  }
}

Expression::Expression(const container_type& container) {
  map.data.emplace(container, complex_type{1.0, 0.0});
}

Expression::Expression(container_type&& container) {
  map.data.emplace(std::move(container), complex_type{1.0, 0.0});
}

Expression::Expression(std::initializer_list<Term> lst) {
  for (const auto& term : lst) {
    add_to_map(map, term.operators, term.c);
  }
}

Expression Expression::adjoint() const {
  Expression result;
  for (const auto& [ops, coeff] : map.data) {
    Term term(coeff, ops);
    Term adj = term.adjoint();
    add_to_map(result.map, std::move(adj.operators), adj.c);
  }
  return result;
}

Expression& Expression::truncate_by_size(size_t max_size) {
  if (max_size == 0) {
    map.clear();
    return *this;
  }
  for (auto it = map.data.begin(); it != map.data.end();) {
    if (it->first.size() > max_size) {
      it = map.data.erase(it);
    } else {
      ++it;
    }
  }
  return *this;
}

Expression& Expression::truncate_by_norm(double min_norm) {
  map.truncate_by_norm(min_norm);
  return *this;
}

Expression& Expression::filter_by_size(size_t size) {
  if (size == 0) {
    map.clear();
    return *this;
  }
  for (auto it = map.data.begin(); it != map.data.end();) {
    if (it->first.size() != size) {
      it = map.data.erase(it);
    } else {
      ++it;
    }
  }
  return *this;
}

void Expression::to_string(std::ostringstream& oss) const {
  map.format_sorted(
      oss, [](std::ostringstream& os, const container_type& ops, const complex_type& coeff) {
        Term term(coeff, ops);
        term.to_string(os);
      });
}

Expression& Expression::operator*=(const Expression& value) {
  if (map.empty() || value.map.empty()) {
    map.clear();
    return *this;
  }
  ExpressionMap<container_type> result;
  result.reserve(map.size() * value.map.size());
  for (const auto& [lhs_ops, lhs_coeff] : map.data) {
    for (const auto& [rhs_ops, rhs_coeff] : value.map.data) {
      if (lhs_ops.size() + rhs_ops.size() > Term::static_vector_size) {
        continue;
      }
      container_type combined = lhs_ops;
      combined.append_range(rhs_ops.begin(), rhs_ops.end());
      add_to_map(result, std::move(combined), lhs_coeff * rhs_coeff);
    }
  }
  map.data = std::move(result.data);
  return *this;
}

Expression& Expression::operator+=(const Term& value) {
  add_to_map(map, value.operators, value.c);
  return *this;
}

Expression& Expression::operator-=(const Term& value) {
  add_to_map(map, value.operators, -value.c);
  return *this;
}

Expression& Expression::operator*=(const Term& value) {
  if (map.empty()) {
    return *this;
  }
  if (ExpressionMap<container_type>::is_zero(value.c)) {
    map.clear();
    return *this;
  }

  if (value.operators.size() == 0) {
    for (auto& [ops, coeff] : map.data) {
      coeff *= value.c;
    }
    return *this;
  }
  ExpressionMap<container_type> result;
  result.reserve(map.size());
  for (const auto& [ops, coeff] : map.data) {
    if (ops.size() + value.operators.size() > Term::static_vector_size) {
      continue;
    }
    container_type combined = ops;
    combined.append_range(value.operators.begin(), value.operators.end());
    add_to_map(result, std::move(combined), coeff * value.c);
  }
  map.data = std::move(result.data);
  return *this;
}
