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

  size_t size() const;

  std::string to_string() const;

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
  static bool is_zero(const complex_type& value);

  static bool less_ops(const container_type& left, const container_type& right);

  static void add_to_map(map_type& target, const container_type& ops, const complex_type& coeff);

  static void add_to_map(map_type& target, container_type&& ops, const complex_type& coeff);
};
