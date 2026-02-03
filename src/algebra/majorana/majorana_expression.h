#pragma once

#include <complex>
#include <sstream>
#include <string>

#include "algebra/majorana/majorana_string.h"
#include "robin_hood.h"

struct MajoranaExpression {
  using complex_type = std::complex<double>;
  using map_type = robin_hood::unordered_map<MajoranaString, complex_type>;

  map_type hashmap{};

  MajoranaExpression() = default;
  ~MajoranaExpression() = default;

  MajoranaExpression(const MajoranaExpression&) = default;
  MajoranaExpression& operator=(const MajoranaExpression&) = default;
  MajoranaExpression(MajoranaExpression&&) noexcept = default;
  MajoranaExpression& operator=(MajoranaExpression&&) noexcept = default;

  explicit MajoranaExpression(complex_type c);
  explicit MajoranaExpression(int sign, const MajoranaString& str);
  explicit MajoranaExpression(complex_type c, const MajoranaString& str);

  size_t size() const;

  double norm_squared() const;

  MajoranaExpression& truncate_by_norm(double min_norm);

  void to_string(std::ostringstream& oss) const;
  std::string to_string() const;

  MajoranaExpression& operator+=(const complex_type& value);
  MajoranaExpression& operator-=(const complex_type& value);
  MajoranaExpression& operator*=(const complex_type& value);
  MajoranaExpression& operator/=(const complex_type& value);

  MajoranaExpression& operator+=(const MajoranaExpression& value);
  MajoranaExpression& operator-=(const MajoranaExpression& value);
  MajoranaExpression& operator*=(const MajoranaExpression& value);

  friend MajoranaExpression commutator(const MajoranaExpression& A, const MajoranaExpression& B);
  friend MajoranaExpression anticommutator(const MajoranaExpression& A,
                                           const MajoranaExpression& B);

  static bool is_zero(const complex_type& value);
  static void add_to_map(map_type& target, const MajoranaString& str, const complex_type& coeff);
  static void add_to_map(map_type& target, MajoranaString&& str, const complex_type& coeff);
};

inline MajoranaExpression operator+(MajoranaExpression lhs, const MajoranaExpression& rhs) {
  lhs += rhs;
  return lhs;
}

inline MajoranaExpression operator-(MajoranaExpression lhs, const MajoranaExpression& rhs) {
  lhs -= rhs;
  return lhs;
}

inline MajoranaExpression operator*(MajoranaExpression lhs, const MajoranaExpression& rhs) {
  lhs *= rhs;
  return lhs;
}

inline MajoranaExpression operator*(MajoranaExpression lhs,
                                    const MajoranaExpression::complex_type& rhs) {
  lhs *= rhs;
  return lhs;
}

inline MajoranaExpression operator*(const MajoranaExpression::complex_type& lhs,
                                    MajoranaExpression rhs) {
  rhs *= lhs;
  return rhs;
}
