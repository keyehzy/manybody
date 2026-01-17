#include "algorithms/schriffer_wolff.h"

#include <complex>

#include "algebra/matrix_elements.h"
#include "utils/tolerances.h"

namespace {
constexpr auto tolerance =
    tolerances::tolerance<Expression::complex_type::value_type>();
}  // namespace

std::pair<size_t, double> cluster_by_largest_gap(const arma::vec& vals) {
  if (vals.n_elem < 2) {
    return {vals.n_elem, 0.0};
  }

  double max_gap = -1.0;
  size_t split_index = 0;
  for (size_t i = 0; i + 1 < vals.n_elem; ++i) {
    double gap = vals(i + 1) - vals(i);
    if (gap > max_gap) {
      max_gap = gap;
      split_index = i;
    }
  }

  return {split_index + 1, max_gap};
}

Expression schriffer_wolff(const Expression& kinetic, const Expression& interaction,
                           const Basis& basis, size_t iter) {
  Expression hamiltonian = kinetic + interaction;

  arma::cx_mat H = compute_matrix_elements<arma::cx_mat>(basis, hamiltonian);
  arma::cx_mat H_interaction = compute_matrix_elements<arma::cx_mat>(basis, interaction);

  arma::vec vals;
  arma::cx_mat vecs;
  if (!arma::eig_sym(vals, vecs, H_interaction)) {
    return {};
  }

  arma::cx_mat Hk = H;
  arma::cx_mat Ufinal = arma::eye<arma::cx_mat>(H.n_rows, H.n_cols);
  const arma::cx_mat& U0 = vecs;
  auto [n, gap] = cluster_by_largest_gap(vals);
  static_cast<void>(gap);

  for (size_t step = 0; step < iter; ++step) {
    bool converged = true;
    arma::cx_mat H_in = U0.t() * Hk * U0;

    arma::cx_mat Atilde(H_in.n_rows, H_in.n_cols, arma::fill::zeros);
    for (size_t i = 0; i < n; ++i) {
      for (size_t j = n; j < H_in.n_rows; ++j) {
        std::complex<double> denom = vals(i) - vals(j);
        if (std::abs(denom) <= tolerance) {
          continue;
        }
        Atilde(i, j) = H_in(i, j) / denom;
        Atilde(j, i) = -std::conj(Atilde(i, j));
      }
    }

    const double max_val = arma::abs(Atilde).max();
    if (max_val > 1e-12) {
      converged = false;
    }

    arma::cx_mat A_sw = U0 * Atilde * U0.t();
    arma::cx_mat expA = arma::expmat(A_sw);
    arma::cx_mat expA_inv = arma::expmat(-A_sw);
    Ufinal *= expA;
    Hk = expA * Hk * expA_inv;

    if (converged) {
      break;
    }
  }

  arma::cx_mat Afinal = arma::logmat(Ufinal);

  Expression Aop;
  for (size_t i = 0; i < basis.set.size(); ++i) {
    for (size_t j = 0; j < basis.set.size(); ++j) {
      const auto coeff = static_cast<Expression::complex_type>(Afinal(j, i));
      if (std::norm(coeff) < tolerance * tolerance) {
        continue;
      }
      Term a = Term(basis.set.at(i));
      Term b = Term(basis.set.at(j));
      Aop += coeff * a * b.adjoint();
    }
  }

  return Aop;
}
