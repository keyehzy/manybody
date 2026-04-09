#pragma once

#include <array>
#include <complex>
#include <sstream>
#include <string>

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
