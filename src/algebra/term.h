#pragma once

#include <array>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <sstream>
#include <string>

#include "algebra/monomial.h"
#include "algebra/operator.h"

constexpr size_t term_size = 32;

using TermScalar = std::complex<double>;

constexpr size_t term_static_vector_size = (term_size - sizeof(TermScalar)) / sizeof(Operator) - 1;

using FermionMonomial =
    MonomialImpl<Operator, term_static_vector_size, Operator::ubyte, TermScalar>;

constexpr bool is_diagonal(const FermionMonomial::container_type& operators) noexcept {
  constexpr size_t stride = Operator::kValueMask + 1;
  std::array<int, stride * 2> balance{};
  for (const auto& op : operators) {
    const size_t idx = static_cast<size_t>(op.data & ~Operator::kTypeBit);
    if (op.type() == Operator::Type::Creation) {
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

constexpr bool is_diagonal(const FermionMonomial& term) noexcept {
  return is_diagonal(term.operators);
}

constexpr FermionMonomial adjoint(const FermionMonomial& term) noexcept {
  FermionMonomial result(std::conj(term.c));
  for (auto it = term.operators.rbegin(); it != term.operators.rend(); ++it) {
    result.operators.push_back(it->adjoint());
  }
  return result;
}

void to_string(std::ostringstream& oss, const FermionMonomial& term);
std::string to_string(const FermionMonomial& term);

inline constexpr FermionMonomial creation(Operator::Spin spin, size_t orbital) noexcept {
  return FermionMonomial({Operator::creation(spin, orbital)});
}

inline constexpr FermionMonomial annihilation(Operator::Spin spin, size_t orbital) noexcept {
  return FermionMonomial({Operator::annihilation(spin, orbital)});
}

inline constexpr FermionMonomial one_body(Operator::Spin s1, size_t o1, Operator::Spin s2,
                                          size_t o2) noexcept {
  return FermionMonomial({Operator::creation(s1, o1), Operator::annihilation(s2, o2)});
}

inline constexpr FermionMonomial two_body(Operator::Spin s1, size_t o1, Operator::Spin s2,
                                          size_t o2, Operator::Spin s3, size_t o3,
                                          Operator::Spin s4, size_t o4) noexcept {
  return FermionMonomial({Operator::creation(s1, o1), Operator::creation(s2, o2),
                          Operator::annihilation(s3, o3), Operator::annihilation(s4, o4)});
}

inline constexpr FermionMonomial density(Operator::Spin s, size_t o) noexcept {
  return FermionMonomial({Operator::creation(s, o), Operator::annihilation(s, o)});
}

inline constexpr FermionMonomial density_density(Operator::Spin s1, size_t i, Operator::Spin s2,
                                                 size_t j) noexcept {
  return density(s1, i) * density(s2, j);
}

static_assert(sizeof(FermionMonomial) == term_size);
