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

using FermionString = static_vector<Operator, term_static_vector_size, Operator::ubyte>;

struct Term : MonomialBase<Term, Operator, term_static_vector_size, Operator::ubyte, TermScalar> {
  using complex_type = TermScalar;
  using container_type = MonomialBase::container_type;

  static constexpr size_t static_vector_size = term_static_vector_size;

  constexpr Term() noexcept = default;
  constexpr ~Term() noexcept = default;

  constexpr Term(const Term& other) noexcept = default;
  constexpr Term& operator=(const Term& other) noexcept = default;
  constexpr Term(Term&& other) noexcept = default;
  constexpr Term& operator=(Term&& other) noexcept = default;

  using MonomialBase::MonomialBase;

  constexpr bool operator==(const Term& other) const noexcept {
    return c == other.c && operators == other.operators;
  }
};

constexpr bool is_diagonal(const FermionString& operators) noexcept {
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

constexpr bool is_diagonal(const Term& term) noexcept { return is_diagonal(term.operators); }

constexpr Term adjoint(const Term& term) noexcept {
  Term result(std::conj(term.c));
  for (auto it = term.operators.rbegin(); it != term.operators.rend(); ++it) {
    result.operators.push_back(it->adjoint());
  }
  return result;
}

void to_string(std::ostringstream& oss, const Term& term);
std::string to_string(const Term& term);

inline constexpr Term operator*(const Term& a, const Term& b) noexcept {
  Term result(a);
  result *= b;
  return result;
}

inline constexpr Term operator*(const Term& a, Operator b) noexcept {
  Term result(a);
  result *= b;
  return result;
}

inline constexpr Term operator*(const Term& a, Term::complex_type b) noexcept {
  Term result(a);
  result *= b;
  return result;
}

inline constexpr Term operator/(const Term& a, Term::complex_type b) noexcept {
  Term result(a);
  result /= b;
  return result;
}

inline constexpr Term operator*(Term::complex_type a, const Term& b) noexcept {
  Term result(a);
  result *= b;
  return result;
}

inline constexpr Term operator*(Operator a, const Term& b) noexcept {
  Term result(a);
  result *= b;
  return result;
}

inline constexpr Term creation(Operator::Spin spin, size_t orbital) noexcept {
  return Term({Operator::creation(spin, orbital)});
}

inline constexpr Term annihilation(Operator::Spin spin, size_t orbital) noexcept {
  return Term({Operator::annihilation(spin, orbital)});
}

inline constexpr Term one_body(Operator::Spin s1, size_t o1, Operator::Spin s2,
                               size_t o2) noexcept {
  return Term({Operator::creation(s1, o1), Operator::annihilation(s2, o2)});
}

inline constexpr Term two_body(Operator::Spin s1, size_t o1, Operator::Spin s2, size_t o2,
                               Operator::Spin s3, size_t o3, Operator::Spin s4,
                               size_t o4) noexcept {
  return Term({Operator::creation(s1, o1), Operator::creation(s2, o2),
               Operator::annihilation(s3, o3), Operator::annihilation(s4, o4)});
}

inline constexpr Term density(Operator::Spin s, size_t o) noexcept {
  return Term({Operator::creation(s, o), Operator::annihilation(s, o)});
}

inline constexpr Term density_density(Operator::Spin s1, size_t i, Operator::Spin s2,
                                      size_t j) noexcept {
  return density(s1, i) * density(s2, j);
}

static_assert(sizeof(Term) == term_size);
