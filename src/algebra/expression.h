#pragma once

#include <algorithm>
#include <sstream>
#include <string>
#include <vector>

#include "algebra/expression_map.h"
#include "algebra/term.h"
#include "robin_hood.h"

struct Expression {
  using complex_type = Term::complex_type;
  using container_type = Term::container_type;
  using map_type = robin_hood::unordered_map<container_type, complex_type>;

  ExpressionMap<container_type> map{};

  const map_type& terms() const { return map.data; }
  map_type& terms() { return map.data; }

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
    map.data.emplace(std::forward<Container>(ops), c);
  }

  Expression adjoint() const;

  size_t size() const;

  Expression& truncate_by_size(size_t max_size);
  Expression& truncate_by_norm(double min_norm);

  Expression& filter_by_size(size_t size);

  void to_string(std::ostringstream& oss) const;
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
  static void add_to_map(map_type& target, const container_type& ops, const complex_type& coeff);

  static void add_to_map(map_type& target, container_type&& ops, const complex_type& coeff);
};

inline Expression operator+(Expression lhs, const Expression& rhs) {
  lhs += rhs;
  return lhs;
}

inline Expression operator-(Expression lhs, const Expression& rhs) {
  lhs -= rhs;
  return lhs;
}

inline Expression operator*(Expression lhs, const Expression& rhs) {
  lhs *= rhs;
  return lhs;
}

inline Expression operator*(Expression lhs, const Expression::complex_type& rhs) {
  lhs *= rhs;
  return lhs;
}

inline Expression operator*(const Expression::complex_type& lhs, Expression rhs) {
  rhs *= lhs;
  return rhs;
}

inline Expression hopping(const Term::complex_type& coeff, size_t from, size_t to,
                          Operator::Spin spin) noexcept {
  Expression result =
      Expression(Term(coeff, {Operator::creation(spin, from), Operator::annihilation(spin, to)}));
  result += Expression(
      Term(std::conj(coeff), {Operator::creation(spin, to), Operator::annihilation(spin, from)}));
  return result;
}

inline Expression hopping(size_t from, size_t to, Operator::Spin spin) noexcept {
  return hopping(1.0, from, to, spin);
}
