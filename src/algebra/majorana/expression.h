#pragma once

#include <complex>
#include <sstream>
#include <string>

#include "algebra/expression_map.h"
#include "algebra/majorana/string.h"
#include "robin_hood.h"

namespace majorana {

struct MajoranaExpression {
  using complex_type = std::complex<double>;
  using container_type = MajoranaMonomial::container_type;
  using map_type = robin_hood::unordered_map<container_type, complex_type>;

  ExpressionMap<container_type> map{};

  MajoranaExpression() = default;
  ~MajoranaExpression() = default;

  MajoranaExpression(const MajoranaExpression&) = default;
  MajoranaExpression& operator=(const MajoranaExpression&) = default;
  MajoranaExpression(MajoranaExpression&&) noexcept = default;
  MajoranaExpression& operator=(MajoranaExpression&&) noexcept = default;

  explicit MajoranaExpression(complex_type c);
  explicit MajoranaExpression(int sign, const container_type& str);
  explicit MajoranaExpression(complex_type c, const container_type& str);
  explicit MajoranaExpression(const MajoranaMonomial& term);

  size_t size() const { return map.size(); }

  double norm_squared() const;

  MajoranaExpression& truncate_by_norm(double min_norm);

  void to_string(std::ostringstream& oss) const;
  std::string to_string() const {
    return map.to_string(
        [](std::ostringstream& os, const container_type& string_data, const complex_type& coeff) {
          os << coeff;
          if (!string_data.empty()) {
            os << " ";
            ::majorana::to_string(os, string_data);
          }
        });
  }

  const map_type& terms() const noexcept { return map.data; }

  MajoranaExpression& operator+=(const complex_type& value) {
    map += value;
    return *this;
  }
  MajoranaExpression& operator-=(const complex_type& value) {
    map -= value;
    return *this;
  }
  MajoranaExpression& operator*=(const complex_type& value) {
    map *= value;
    return *this;
  }
  MajoranaExpression& operator/=(const complex_type& value) {
    map /= value;
    return *this;
  }

  MajoranaExpression& operator+=(const MajoranaExpression& value) {
    map += value.map;
    return *this;
  }
  MajoranaExpression& operator-=(const MajoranaExpression& value) {
    map -= value.map;
    return *this;
  }
  MajoranaExpression& operator*=(const MajoranaExpression& value);

  MajoranaExpression& operator+=(const MajoranaMonomial& value);
  MajoranaExpression& operator-=(const MajoranaMonomial& value);
  MajoranaExpression& operator*=(const MajoranaMonomial& value);

 private:
  static void add_to_map(ExpressionMap<container_type>& target, const container_type& str,
                         const complex_type& coeff);
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

}  // namespace majorana
