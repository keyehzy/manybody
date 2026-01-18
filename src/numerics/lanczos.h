#pragma once

#include <algorithm>
#include <armadillo>
#include <cmath>
#include <limits>
#include <optional>
#include <stdexcept>
#include <tuple>
#include <vector>

#include "numerics/linear_operator.h"
#include "utils/tolerances.h"

template <typename Op>
std::tuple<scalar_real_t<typename Op::ScalarType>,
           std::optional<scalar_real_t<typename Op::ScalarType>>>
lanczos_recurrence_step(const Op& op, typename Op::VectorType& w,
                        const typename Op::VectorType& v_curr,
                        const typename Op::VectorType& v_prev,
                        scalar_real_t<typename Op::ScalarType> beta_prev) {
  using ScalarType = typename Op::ScalarType;
  using RealType = scalar_real_t<ScalarType>;

  w = op.apply(v_curr);

  if (beta_prev != static_cast<RealType>(0)) {
    w -= static_cast<ScalarType>(beta_prev) * v_prev;
  }

  const RealType alpha = std::real(arma::cdot(v_curr, w));
  w -= static_cast<ScalarType>(alpha) * v_curr;

  const RealType beta = arma::norm(w);
  if (beta <= tolerances::tolerance<RealType>()) {
    return {alpha, std::nullopt};
  }

  return {alpha, beta};
}

template <typename Op>
void lanczos_reconstruction_step(const Op& op, typename Op::VectorType& w,
                                 const typename Op::VectorType& v_curr,
                                 const typename Op::VectorType& v_prev,
                                 scalar_real_t<typename Op::ScalarType> alpha_j,
                                 scalar_real_t<typename Op::ScalarType> beta_prev) {
  using ScalarType = typename Op::ScalarType;
  using RealType = scalar_real_t<ScalarType>;

  w = op.apply(v_curr);

  if (beta_prev != static_cast<RealType>(0)) {
    w -= static_cast<ScalarType>(beta_prev) * v_prev;
  }

  w -= static_cast<ScalarType>(alpha_j) * v_curr;
}

template <typename Scalar>
struct LanczosDecomposition {
  using RealType = scalar_real_t<Scalar>;

  std::vector<RealType> alphas{};
  std::vector<RealType> betas{};
  size_t steps_taken = 0;
  RealType b_norm = static_cast<RealType>(0);
};

template <typename Op>
struct LanczosIteration {
  using VectorType = typename Op::VectorType;
  using ScalarType = typename Op::ScalarType;
  using RealType = scalar_real_t<ScalarType>;

  struct Step {
    RealType alpha;
    RealType beta;
  };

  const Op& op;
  VectorType v_prev;
  VectorType v_curr;
  VectorType work;
  RealType beta_prev;
  size_t step;
  size_t max_steps;

  static LanczosIteration create(const Op& op, const VectorType& b, size_t max_steps,
                                 RealType b_norm) {
    if (b_norm <= tolerances::tolerance<RealType>()) {
      throw std::runtime_error("Input vector has zero norm");
    }

    VectorType v_prev(b.n_elem, arma::fill::zeros);
    VectorType v_curr = b / static_cast<ScalarType>(b_norm);
    VectorType work(b.n_elem, arma::fill::zeros);

    return LanczosIteration{
        op, std::move(v_prev), std::move(v_curr), std::move(work), static_cast<RealType>(0),
        0,  max_steps};
  }

  std::optional<Step> next() {
    if (step >= max_steps) {
      return std::nullopt;
    }

    auto [alpha, beta_opt] = lanczos_recurrence_step(op, work, v_curr, v_prev, beta_prev);

    step += 1;

    if (beta_opt) {
      const RealType beta = *beta_opt;

      work /= static_cast<ScalarType>(beta);

      std::swap(v_prev, v_curr);
      std::swap(v_curr, work);
      beta_prev = beta;

      return Step{alpha, beta};
    }

    step = max_steps;
    return Step{alpha, static_cast<RealType>(0)};
  }
};

template <typename Op>
LanczosDecomposition<typename Op::ScalarType> lanczos_pass_one(const Op& op,
                                                               const typename Op::VectorType& b,
                                                               size_t k) {
  using ScalarType = typename Op::ScalarType;
  using RealType = scalar_real_t<ScalarType>;

  if (op.dimension() != static_cast<size_t>(b.n_elem)) {
    throw std::runtime_error("Operator dimension does not match input vector");
  }

  LanczosDecomposition<ScalarType> decomp;
  decomp.b_norm = arma::norm(b);
  if (decomp.b_norm <= tolerances::tolerance<RealType>()) {
    throw std::runtime_error("Input vector has zero norm");
  }

  if (k == 0) {
    return decomp;
  }

  decomp.alphas.reserve(k);
  decomp.betas.reserve(k > 0 ? k - 1 : 0);

  auto it = LanczosIteration<Op>::create(op, b, k, decomp.b_norm);

  for (size_t i = 0; i < k; ++i) {
    const auto step = it.next();
    if (!step) {
      break;
    }

    decomp.alphas.push_back(step->alpha);
    decomp.steps_taken += 1;

    if (step->beta <= tolerances::tolerance<RealType>()) {
      break;
    }

    if (i + 1 < k) {
      decomp.betas.push_back(step->beta);
    }
  }

  return decomp;
}

template <typename Op, typename Scalar>
typename Op::VectorType lanczos_pass_two(
    const Op& op, const typename Op::VectorType& b,
    const LanczosDecomposition<typename Op::ScalarType>& decomp, const std::vector<Scalar>& y_k) {
  using VectorType = typename Op::VectorType;
  using ScalarType = typename Op::ScalarType;
  using RealType = scalar_real_t<ScalarType>;

  if (decomp.steps_taken != y_k.size()) {
    throw std::runtime_error("Dimension mismatch: y_k size must match decomposition steps");
  }

  if (decomp.b_norm <= tolerances::tolerance<RealType>()) {
    throw std::runtime_error("Input vector has zero norm");
  }

  if (decomp.steps_taken == 0) {
    return VectorType(b.n_elem, arma::fill::zeros);
  }

  VectorType v_prev(b.n_elem, arma::fill::zeros);
  VectorType v_curr = b / static_cast<ScalarType>(decomp.b_norm);

  VectorType x_k = v_curr * static_cast<ScalarType>(y_k[0]);
  VectorType work(b.n_elem, arma::fill::zeros);

  for (size_t j = 0; j < decomp.steps_taken - 1; ++j) {
    const RealType alpha_j = decomp.alphas[j];
    const RealType beta_j = decomp.betas[j];
    const RealType beta_prev = (j == 0) ? static_cast<RealType>(0) : decomp.betas[j - 1];

    lanczos_reconstruction_step(op, work, v_curr, v_prev, alpha_j, beta_prev);
    work /= static_cast<ScalarType>(beta_j);

    x_k += work * static_cast<ScalarType>(y_k[j + 1]);

    std::swap(v_prev, v_curr);
    std::swap(v_curr, work);
  }

  return x_k;
}

template <typename Op, typename Solver>
typename Op::VectorType solve(const Op& op, const typename Op::VectorType& b, size_t k,
                              Solver&& solver) {
  using VectorType = typename Op::VectorType;

  const auto decomp = lanczos_pass_one(op, b, k);
  if (decomp.steps_taken == 0) {
    return VectorType(b.n_elem, arma::fill::zeros);
  }

  auto y_k = solver(decomp.alphas, decomp.betas);

  for (auto& value : y_k) {
    value *= decomp.b_norm;
  }

  return lanczos_pass_two<Op, typename decltype(y_k)::value_type>(op, b, decomp, y_k);
}

template <typename Scalar>
struct EigenPair {
  using RealType = scalar_real_t<Scalar>;
  RealType value;
  arma::Col<Scalar> vector;
};

template <typename Op>
EigenPair<typename Op::ScalarType> find_max_eigenpair(const Op& op, size_t k) {
  using ScalarType = typename Op::ScalarType;
  using RealType = scalar_real_t<ScalarType>;

  auto seed = make_seed_vector(op);
  auto decomp = lanczos_pass_one(op, seed, k);
  const size_t m = decomp.steps_taken;
  if (m == 0) {
    return {static_cast<RealType>(0), typename Op::VectorType(op.dimension(), arma::fill::zeros)};
  }

  arma::Mat<RealType> T(m, m, arma::fill::zeros);
  for (size_t i = 0; i < m; ++i) {
    T(i, i) = decomp.alphas[i];
    if (i + 1 < m) {
      T(i, i + 1) = decomp.betas[i];
      T(i + 1, i) = decomp.betas[i];
    }
  }

  arma::Col<RealType> vals;
  arma::Mat<RealType> vecs;
  if (!arma::eig_sym(vals, vecs, T)) {
    throw std::runtime_error("Eigenvalue decomposition of T_k failed");
  }

  const RealType max_val = vals(m - 1);
  std::vector<RealType> y_k(m);
  for (size_t i = 0; i < m; ++i) {
    y_k[i] = vecs(i, m - 1) * decomp.b_norm;
  }

  auto max_vec = lanczos_pass_two(op, seed, decomp, y_k);
  return {max_val, std::move(max_vec)};
}

template <typename Op>
EigenPair<typename Op::ScalarType> find_min_eigenpair(const Op& op, size_t k) {
  Negated<Op> neg_op(op);
  auto result = find_max_eigenpair(neg_op, k);

  result.value = -result.value;
  return result;
}
