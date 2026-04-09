#pragma once

#include <array>
#include <complex>
#include <cstddef>
#include <sstream>
#include <string>

#include "algebra/operator.h"
#include "algebra/monomial.h"

using BosonScalar = std::complex<double>;

constexpr size_t boson_term_size = 32;

constexpr size_t boson_term_static_vector_size =
    (boson_term_size - sizeof(BosonScalar)) / sizeof(BosonOperator) - 1;

using BosonMonomial =
    MonomialImpl<BosonOperator, boson_term_static_vector_size, BosonOperator::ubyte, BosonScalar>;

constexpr bool is_diagonal(const BosonMonomial::container_type& operators) noexcept {
  constexpr size_t stride = BosonOperator::kValueMask + 1;
  std::array<int, stride * 2> balance{};
  for (const auto& op : operators) {
    const size_t idx = static_cast<size_t>(op.data & ~BosonOperator::kTypeBit);
    if (op.type() == BosonOperator::Type::Creation) {
      ++balance[idx];
    } else {
      --balance[idx];
    }
  }
  for (int count : balance) {
    if (count != 0) {
      return false;
    }
  }
  return true;
}

constexpr bool is_diagonal(const BosonMonomial& term) noexcept {
  return is_diagonal(term.operators);
}

constexpr BosonMonomial adjoint(const BosonMonomial& term) noexcept {
  BosonMonomial result(std::conj(term.c));
  for (auto it = term.operators.rbegin(); it != term.operators.rend(); ++it) {
    result.operators.push_back(it->adjoint());
  }
  return result;
}

void to_string(std::ostringstream& oss, const BosonMonomial& term);
std::string to_string(const BosonMonomial& term);

namespace boson {

inline constexpr BosonMonomial creation(BosonOperator::Spin spin, size_t orbital) noexcept {
  return BosonMonomial({BosonOperator::creation(spin, orbital)});
}

inline constexpr BosonMonomial annihilation(BosonOperator::Spin spin, size_t orbital) noexcept {
  return BosonMonomial({BosonOperator::annihilation(spin, orbital)});
}

inline constexpr BosonMonomial one_body(BosonOperator::Spin s1, size_t o1, BosonOperator::Spin s2,
                                        size_t o2) noexcept {
  return BosonMonomial({BosonOperator::creation(s1, o1), BosonOperator::annihilation(s2, o2)});
}

inline constexpr BosonMonomial two_body(BosonOperator::Spin s1, size_t o1, BosonOperator::Spin s2,
                                        size_t o2, BosonOperator::Spin s3, size_t o3,
                                        BosonOperator::Spin s4, size_t o4) noexcept {
  return BosonMonomial({BosonOperator::creation(s1, o1), BosonOperator::creation(s2, o2),
                        BosonOperator::annihilation(s3, o3), BosonOperator::annihilation(s4, o4)});
}

inline constexpr BosonMonomial number_op(BosonOperator::Spin s, size_t o) noexcept {
  return BosonMonomial({BosonOperator::creation(s, o), BosonOperator::annihilation(s, o)});
}

}  // namespace boson

static_assert(sizeof(BosonMonomial) == boson_term_size);
