#pragma once

#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <numbers>

#include "algebra/model/model.h"
#include "algebra/term.h"

namespace detail {
inline size_t canonicalize_momentum(int64_t value, size_t lattice_size) {
  if (lattice_size == 0) {
    return 0;
  }
  const int64_t L = static_cast<int64_t>(lattice_size);
  int64_t mod = value % L;
  if (mod < 0) {
    mod += L;
  }
  return static_cast<size_t>(mod);
}
}  // namespace detail

// Two-particle Hubbard model in the relative coordinate basis with fixed total momentum K.
// B^+_{K,p} = c^+_{p,up} c^+_{K-p,down}
// B^+_{K,r} = (1/sqrt(N)) sum_p exp(i p r) B^+_{K,p}
// H(K) = t_eff(K) * sum_r (B^+_{K,r} B_{K,r+1} + B^+_{K,r+1} B_{K,r}) + U B^+_{K,0} B_{K,0}
struct HubbardModelRelative : Model {
  HubbardModelRelative(double t, double u, size_t size, int64_t total_momentum)
      : t(t),
        u(u),
        size(size),
        total_momentum(detail::canonicalize_momentum(total_momentum, size)) {}

  double effective_hopping() const {
    const double k_phase = 2.0 * std::numbers::pi_v<double> * static_cast<double>(total_momentum) /
                           static_cast<double>(size);
    return 2.0 * t * std::cos(0.5 * k_phase);
  }

  Expression pair_creation(size_t r) const {
    Expression result;
    const double normalization = 1.0 / std::sqrt(static_cast<double>(size));
    for (size_t p = 0; p < size; ++p) {
      const size_t k_minus_p = (total_momentum + size - (p % size)) % size;
      const double phase =
          2.0 * std::numbers::pi_v<double> * static_cast<double>(p * r) / static_cast<double>(size);
      const std::complex<double> coefficient =
          std::exp(std::complex<double>(0.0, phase)) * normalization;
      result += Term(Term::complex_type(static_cast<float>(coefficient.real()),
                                        static_cast<float>(coefficient.imag())),
                     {Operator::creation(Operator::Spin::Up, p),
                      Operator::creation(Operator::Spin::Down, k_minus_p)});
    }
    return result;
  }

  Expression pair_annihilation(size_t r) const { return pair_creation(r).adjoint(); }

  Expression kinetic() const {
    Expression kinetic_term;
    const double t_eff = effective_hopping();
    for (size_t r = 0; r < size; ++r) {
      const size_t next = (r + 1) % size;
      kinetic_term += t_eff * (pair_creation(r) * pair_annihilation(next));
      kinetic_term += t_eff * (pair_creation(next) * pair_annihilation(r));
    }
    return kinetic_term;
  }

  Expression interaction() const {
    Expression interaction_term;
    interaction_term += u * (pair_creation(0) * pair_annihilation(0));
    return interaction_term;
  }

  Expression hamiltonian() const override {
    Expression result = kinetic();
    result += interaction();
    return result;
  }

  double t;
  double u;
  size_t size;
  size_t total_momentum;
};
