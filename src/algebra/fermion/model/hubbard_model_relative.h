#pragma once

#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <numbers>

#include "algebra/fermion/model/model.h"
#include "algebra/fermion/term.h"
#include "utils/canonicalize_momentum.h"

// Two-particle Hubbard model in the relative coordinate basis with fixed total momentum K.
// B^+_{K,p} = c^+_{p,up} c^+_{K-p,down}
// B^+_{K,r} = (1/sqrt(N)) sum_p exp(i p r) B^+_{K,p}
// H(K) = t_eff(K) * sum_r (e^{ik/2} B^+_{K,r} B_{K,r+1} + e^{-ik/2} B^+_{K,r+1} B_{K,r})
//        + U B^+_{K,0} B_{K,0},   k = 2 pi K / N.
// The phase reproduces the exact pair dispersion eps(p) + eps(K-p) = 2 t_eff cos(p - k/2) in
// this (periodic) gauge; without it the spectrum is wrong for every K with t_eff != 0 and
// k/2 not a multiple of 2 pi / N.
struct HubbardModelRelative : FermionModel {
  HubbardModelRelative(double t_val, double u_val, size_t size_val, int64_t total_momentum_val)
      : t(t_val),
        u(u_val),
        size(size_val),
        total_momentum(utils::canonicalize_momentum(total_momentum_val, size_val)) {}

  double effective_hopping() const {
    const double k_phase = 2.0 * std::numbers::pi_v<double> * static_cast<double>(total_momentum) /
                           static_cast<double>(size);
    return -2.0 * t * std::cos(0.5 * k_phase);
  }

  FermionExpression pair_creation(size_t r) const {
    FermionExpression result;
    const double normalization = 1.0 / std::sqrt(static_cast<double>(size));
    for (size_t p = 0; p < size; ++p) {
      const size_t k_minus_p = (total_momentum + size - (p % size)) % size;
      const double phase =
          2.0 * std::numbers::pi_v<double> * static_cast<double>(p * r) / static_cast<double>(size);
      const std::complex<double> coefficient =
          std::exp(std::complex<double>(0.0, phase)) * normalization;
      result +=
          FermionMonomial(FermionMonomial::complex_type(coefficient.real(), coefficient.imag()),
                          {FermionOperator::creation(FermionOperator::Spin::Up, p),
                           FermionOperator::creation(FermionOperator::Spin::Down, k_minus_p)});
    }
    return result;
  }

  FermionExpression pair_annihilation(size_t r) const { return adjoint(pair_creation(r)); }

  FermionExpression kinetic() const {
    FermionExpression kinetic_term;
    const double half_k = std::numbers::pi_v<double> * static_cast<double>(total_momentum) /
                          static_cast<double>(size);
    const std::complex<double> forward = effective_hopping() * std::polar(1.0, half_k);
    const auto forward_coeff = FermionExpression::complex_type(forward.real(), forward.imag());
    const auto backward_coeff = FermionExpression::complex_type(forward.real(), -forward.imag());
    for (size_t r = 0; r < size; ++r) {
      const size_t next = (r + 1) % size;
      kinetic_term += forward_coeff * (pair_creation(r) * pair_annihilation(next));
      kinetic_term += backward_coeff * (pair_creation(next) * pair_annihilation(r));
    }
    return kinetic_term;
  }

  FermionExpression interaction() const {
    FermionExpression interaction_term;
    const auto u_coeff = FermionExpression::complex_type(u, 0.0);
    interaction_term += u_coeff * (pair_creation(0) * pair_annihilation(0));
    return interaction_term;
  }

  FermionExpression hamiltonian() const override {
    FermionExpression result = kinetic();
    result += interaction();
    return result;
  }

  double t;
  double u;
  size_t size;
  size_t total_momentum;
};
