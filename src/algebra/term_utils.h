#pragma once

#include <array>
#include <complex>
#include <cstddef>
#include <sstream>
#include <string>

namespace detail {

template <typename Monomial, typename... Operators>
inline constexpr Monomial make_monomial(Operators... operators) noexcept {
  return Monomial({operators...});
}

template <typename Monomial>
inline constexpr Monomial make_creation_monomial(typename Monomial::operator_type::Spin spin,
                                                 size_t orbital) noexcept {
  using operator_type = typename Monomial::operator_type;
  return make_monomial<Monomial>(operator_type::creation(spin, orbital));
}

template <typename Monomial>
inline constexpr Monomial make_annihilation_monomial(typename Monomial::operator_type::Spin spin,
                                                     size_t orbital) noexcept {
  using operator_type = typename Monomial::operator_type;
  return make_monomial<Monomial>(operator_type::annihilation(spin, orbital));
}

template <typename Monomial>
inline constexpr Monomial make_one_body_monomial(typename Monomial::operator_type::Spin s1,
                                                 size_t o1,
                                                 typename Monomial::operator_type::Spin s2,
                                                 size_t o2) noexcept {
  using operator_type = typename Monomial::operator_type;
  return make_monomial<Monomial>(operator_type::creation(s1, o1),
                                 operator_type::annihilation(s2, o2));
}

template <typename Monomial>
inline constexpr Monomial make_two_body_monomial(
    typename Monomial::operator_type::Spin s1, size_t o1, typename Monomial::operator_type::Spin s2,
    size_t o2, typename Monomial::operator_type::Spin s3, size_t o3,
    typename Monomial::operator_type::Spin s4, size_t o4) noexcept {
  using operator_type = typename Monomial::operator_type;
  return make_monomial<Monomial>(operator_type::creation(s1, o1), operator_type::creation(s2, o2),
                                 operator_type::annihilation(s3, o3),
                                 operator_type::annihilation(s4, o4));
}

template <typename Monomial>
inline constexpr Monomial make_number_monomial(typename Monomial::operator_type::Spin spin,
                                               size_t orbital) noexcept {
  using operator_type = typename Monomial::operator_type;
  return make_monomial<Monomial>(operator_type::creation(spin, orbital),
                                 operator_type::annihilation(spin, orbital));
}

}  // namespace detail

template <typename Monomial>
constexpr bool is_diagonal(const Monomial& term) noexcept {
  using op_type = typename Monomial::operator_type;
  constexpr size_t stride = op_type::kValueMask + 1;
  std::array<int, stride * 2> balance{};
  for (const auto& op : term.operators) {
    const size_t idx = static_cast<size_t>(op.data & ~op_type::kTypeBit);
    if (op.type() == op_type::Type::Creation) {
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

template <typename Monomial>
constexpr Monomial adjoint(const Monomial& term) noexcept {
  Monomial result(std::conj(term.c));
  for (auto it = term.operators.rbegin(); it != term.operators.rend(); ++it) {
    result.operators.push_back(it->adjoint());
  }
  return result;
}

template <typename Monomial>
void to_string(std::ostringstream& oss, const Monomial& term) {
  if (term.c == typename Monomial::complex_type{}) {
    oss << "0";
    return;
  }
  oss << term.c;
  if (term.operators.size() == 0) {
    return;
  }
  oss << " ";
  for (const auto& op : term.operators) {
    op.to_string(oss);
  }
  if (is_diagonal(term)) {
    oss << "*";
  }
}

template <typename Monomial>
std::string to_string(const Monomial& term) {
  std::ostringstream oss;
  to_string(oss, term);
  return oss.str();
}
