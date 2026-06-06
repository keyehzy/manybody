#pragma once

#include <array>
#include <complex>
#include <cstddef>
#include <utility>

#include "algebra/fermion/expression.h"
#include "algebra/operator.h"
#include "algebra/reference_state.h"
#include "utils/static_vector.h"

// Generalised Wick decoupling against a (possibly non-trivial) reference state.
//
// Standard `canonicalize()` performs normal ordering against the vacuum: the
// only non-zero contraction is <c c+>_vac = 1. The Bartlett self-consistent
// linearisation scheme (Scheme B, applied to the Hubbard model) requires
// contractions against a flowing reference state characterised by the
// one-body density n_k(l) and an irreducible double-occupancy cumulant D(l).
//
// `wick_decouple` performs the Gaussian Wick decomposition
//
//     A_1 ... A_{2n} = sum_{S over pair partitions of {1..2n}}
//                          sign(S) * prod_{(i,j) in S} <A_i A_j>_ref
//                        * N{ unpaired A's in original order }
//
// where N{...} is the operator string sorted with Pauli signs but no further
// contractions ("pauli_sort"). Including every subset of pairings (including
// the empty subset) makes this an operator identity for any quasi-free
// reference; the non-Gaussian double-occupancy cumulant is handled separately
// by `wick_expectation` via DiagonalFermionReference::double_occupancy_cumulant.

namespace detail {

inline FermionExpression pauli_sort(FermionMonomial::container_type ops,
                                    FermionExpression::complex_type coeff) {
  if (FermionExpression::is_zero(coeff)) {
    return {};
  }
  if (ops.size() < 2) {
    if (ops.size() == 0) {
      FermionExpression result;
      result.add(FermionMonomial::container_type{}, coeff);
      return result;
    }
    return FermionExpression(coeff, std::move(ops));
  }
  if (has_consecutive_elements(ops)) {
    return {};
  }

  FermionExpression::complex_type phase{1.0, 0.0};
  for (std::size_t i = 1; i < ops.size(); ++i) {
    std::size_t j = i;
    while (j > 0 && ops[j] < ops[j - 1]) {
      std::swap(ops[j], ops[j - 1]);
      phase = -phase;
      --j;
    }
  }

  if (has_consecutive_elements(ops)) {
    return {};
  }

  FermionExpression result;
  result.add(std::move(ops), coeff * phase);
  return result;
}

template <typename Ref>
FermionExpression wick_decouple_recursive(const FermionMonomial::container_type& ops,
                                          const Ref& ref) {
  using complex_type = FermionExpression::complex_type;
  using Type = FermionOperator::Type;

  if (ops.size() == 0) {
    FermionExpression result;
    result.add(FermionMonomial::container_type{}, complex_type{1.0, 0.0});
    return result;
  }

  FermionExpression result;

  FermionMonomial::container_type tail;
  for (std::size_t i = 1; i < ops.size(); ++i) {
    tail.push_back(ops[i]);
  }
  {
    FermionExpression rest = wick_decouple_recursive(tail, ref);
    for (const auto& [tail_ops, tail_coeff] : rest.terms()) {
      FermionMonomial::container_type new_ops;
      new_ops.push_back(ops[0]);
      for (const auto& op : tail_ops) {
        new_ops.push_back(op);
      }
      result.add(std::move(new_ops), tail_coeff);
    }
  }

  for (std::size_t j = 1; j < ops.size(); ++j) {
    complex_type contraction{0.0, 0.0};
    const FermionOperator& a = ops[0];
    const FermionOperator& b = ops[j];
    if (a.type() == Type::Creation && b.type() == Type::Annihilation) {
      contraction = ref.contract_dagger_then_plain(a, b);
    } else if (a.type() == Type::Annihilation && b.type() == Type::Creation) {
      contraction = ref.contract_plain_then_dagger(a, b);
    } else if (a.type() == Type::Annihilation && b.type() == Type::Annihilation) {
      contraction = ref.contract_plain_then_plain(a, b);
    } else {
      contraction = ref.contract_dagger_then_dagger(a, b);
    }

    if (FermionExpression::is_zero(contraction)) {
      continue;
    }

    const double sign = ((j - 1) % 2 == 0) ? 1.0 : -1.0;
    const complex_type scalar = contraction * complex_type{sign, 0.0};

    FermionMonomial::container_type residual;
    for (std::size_t k = 1; k < ops.size(); ++k) {
      if (k != j) {
        residual.push_back(ops[k]);
      }
    }

    FermionExpression sub = wick_decouple_recursive(residual, ref);
    sub *= scalar;
    result += sub;
  }

  return result;
}

}  // namespace detail

template <typename Ref>
FermionExpression wick_decouple(const FermionMonomial& m, const Ref& ref) {
  using complex_type = FermionExpression::complex_type;
  if (FermionExpression::is_zero(m.c)) {
    return {};
  }
  FermionExpression raw = detail::wick_decouple_recursive(m.operators, ref);
  FermionExpression result;
  for (const auto& [ops, coeff] : raw.terms()) {
    FermionMonomial::container_type ops_copy = ops;
    FermionExpression sorted =
        detail::pauli_sort(std::move(ops_copy), static_cast<complex_type>(coeff * m.c));
    result += sorted;
  }
  return result;
}

template <typename Ref>
FermionExpression wick_decouple(const FermionExpression& e, const Ref& ref) {
  FermionExpression result;
  for (const auto& [ops, coeff] : e.terms()) {
    FermionMonomial m(coeff, ops);
    result += wick_decouple(m, ref);
  }
  return result;
}

namespace detail {

template <typename Ref>
std::complex<double> on_site_double_occupancy_correction(const FermionMonomial& m, const Ref& ref) {
  using Type = FermionOperator::Type;
  using Spin = FermionOperator::Spin;

  if (m.operators.size() != 4) {
    return {0.0, 0.0};
  }

  int idx_up_dag = -1, idx_up = -1, idx_dn_dag = -1, idx_dn = -1;
  std::size_t site = 0;
  bool site_set = false;

  for (std::size_t i = 0; i < 4; ++i) {
    const auto& op = m.operators[i];
    if (!site_set) {
      site = op.value();
      site_set = true;
    } else if (op.value() != site) {
      return {0.0, 0.0};
    }
    if (op.spin() == Spin::Up) {
      if (op.type() == Type::Creation) {
        if (idx_up_dag != -1) return {0.0, 0.0};
        idx_up_dag = static_cast<int>(i);
      } else {
        if (idx_up != -1) return {0.0, 0.0};
        idx_up = static_cast<int>(i);
      }
    } else {
      if (op.type() == Type::Creation) {
        if (idx_dn_dag != -1) return {0.0, 0.0};
        idx_dn_dag = static_cast<int>(i);
      } else {
        if (idx_dn != -1) return {0.0, 0.0};
        idx_dn = static_cast<int>(i);
      }
    }
  }
  if (idx_up_dag < 0 || idx_up < 0 || idx_dn_dag < 0 || idx_dn < 0) {
    return {0.0, 0.0};
  }

  std::array<int, 4> perm = {idx_up_dag, idx_up, idx_dn_dag, idx_dn};
  int inv = 0;
  for (int i = 0; i < 4; ++i) {
    for (int j = i + 1; j < 4; ++j) {
      if (perm[i] > perm[j]) ++inv;
    }
  }
  const double sign = (inv % 2 == 0) ? 1.0 : -1.0;
  return m.c * std::complex<double>(sign, 0.0) * ref.double_occupancy_cumulant(site);
}

}  // namespace detail

template <typename Ref>
std::complex<double> wick_expectation(const FermionMonomial& m, const Ref& ref) {
  if (FermionExpression::is_zero(m.c)) {
    return {0.0, 0.0};
  }
  FermionExpression decoupled = wick_decouple(m, ref);
  const FermionMonomial::container_type empty{};
  auto it = decoupled.terms().find(empty);
  std::complex<double> gaussian =
      (it == decoupled.terms().end()) ? std::complex<double>{0.0, 0.0} : it->second;
  std::complex<double> cumulant = detail::on_site_double_occupancy_correction(m, ref);
  return gaussian + cumulant;
}

template <typename Ref>
std::complex<double> wick_expectation(const FermionExpression& e, const Ref& ref) {
  std::complex<double> result{0.0, 0.0};
  for (const auto& [ops, coeff] : e.terms()) {
    FermionMonomial m(coeff, ops);
    result += wick_expectation(m, ref);
  }
  return result;
}
