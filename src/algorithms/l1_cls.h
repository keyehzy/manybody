#pragma once

#include <armadillo>
#include <cmath>
#include <set>
#include <vector>

/// L1-norm minimization for extracting Compact Localized States (CLS).
///
/// Given a Hamiltonian H with a degenerate eigenvalue E_target, finds the
/// sparsest eigenstates by minimizing the L1-norm of psi = Z * c over the
/// null-space basis Z, using Iteratively Reweighted Least Squares (IRLS)
/// with geometric eps-annealing.
namespace l1_cls {

struct IrlsParams {
  size_t max_iter_per_stage = 50;
  double eps_start = 1e-2;
  double eps_end = 1e-14;
  size_t n_eps_stages = 7;
  double convergence_tol = 1e-12;
  double cleanup_tol = 1e-10;
  double svd_tol = 1e-10;
};

struct DiagnosticResult {
  size_t nullspace_dim;
  size_t n_cls_found;
  size_t cls_rank;
  size_t incompleteness_gap;
  std::vector<arma::vec> cls_list;
};

/// Extract an orthonormal basis for the null space of (H - E_target * I).
///
/// Returns Z whose columns span ker(H - E_target * I), identified via SVD
/// as the right singular vectors whose singular values fall below svd_tol.
inline arma::mat compute_nullspace(const arma::mat& H, double E_target,
                                   double svd_tol = 1e-10) {
  const size_t dim = H.n_rows;
  const arma::mat M = H - E_target * arma::eye(dim, dim);

  arma::mat U;
  arma::vec s;
  arma::mat V;
  arma::svd(U, s, V, M);

  // Count null singular values
  size_t null_count = 0;
  for (size_t i = 0; i < s.n_elem; ++i) {
    if (s(i) < svd_tol) {
      ++null_count;
    }
  }

  // The null space vectors are the last null_count columns of V
  // (corresponding to the smallest singular values, which svd returns in
  // descending order)
  arma::mat Z = V.cols(V.n_cols - null_count, V.n_cols - 1);
  return Z;
}

/// Phase-convention cleanup: make the largest-magnitude component positive,
/// threshold numerical noise, and re-normalize.
inline arma::vec clean_and_normalize(arma::vec psi, double cleanup_tol) {
  double norm = arma::norm(psi);
  if (norm < 1e-15) {
    return psi;
  }
  psi /= norm;

  // Phase convention: largest-magnitude component positive
  const arma::uword idx_max = arma::index_max(arma::abs(psi));
  if (psi(idx_max) < 0.0) {
    psi = -psi;
  }

  // Zero out noise
  psi.elem(arma::find(arma::abs(psi) < cleanup_tol)).zeros();

  // Re-normalize
  norm = arma::norm(psi);
  if (norm > 0.0) {
    psi /= norm;
  }
  return psi;
}

/// Core IRLS loop with geometric eps-annealing.
///
/// Finds the coefficient vector c that minimizes ||Z * c||_1 subject to
/// ||c|| = 1, by iteratively solving a weighted eigenvalue problem.
/// Returns {c, total_iterations}.
inline std::pair<arma::vec, size_t> irls_annealed(const arma::mat& Z, arma::vec c,
                                                   const IrlsParams& p) {
  const arma::vec eps_schedule = arma::logspace(std::log10(p.eps_start),
                                                std::log10(p.eps_end),
                                                p.n_eps_stages);
  size_t total_iter = 0;

  for (size_t stage = 0; stage < p.n_eps_stages; ++stage) {
    const double eps = eps_schedule(stage);
    const double eps_sq = eps * eps;

    for (size_t iter = 0; iter < p.max_iter_per_stage; ++iter) {
      const arma::vec psi = Z * c;

      // Reweighting: w_i = 1 / sqrt(|psi_i|^2 + eps^2)
      const arma::vec w = 1.0 / arma::sqrt(arma::square(psi) + eps_sq);

      // Weighted Gram matrix: G = Z^T diag(w) Z
      const arma::mat G = Z.t() * arma::diagmat(w) * Z;

      // New c = eigenvector of G with smallest eigenvalue
      arma::vec eigval;
      arma::mat eigvec;
      arma::eig_sym(eigval, eigvec, G);
      arma::vec c_new = eigvec.col(0);

      // Phase alignment: fix sign ambiguity using overlap with previous c
      if (arma::dot(c, c_new) < 0.0) {
        c_new = -c_new;
      }

      ++total_iter;
      const double delta = arma::norm(c_new - c);
      c = c_new;

      if (delta < p.convergence_tol) {
        break;
      }
    }
  }

  return {c, total_iter};
}

/// Find all unique CLS by seeding IRLS from every site.
///
/// For each site index, projects a delta vector onto the null space to obtain
/// a biased starting vector, then runs annealed IRLS. Results are
/// deduplicated by their support set (the set of site indices with nonzero
/// amplitude).
inline std::vector<arma::vec> find_all_cls(const arma::mat& Z, const IrlsParams& p) {
  const size_t n_sites = Z.n_rows;

  std::set<std::vector<arma::uword>> seen_supports;
  std::vector<arma::vec> cls_list;

  for (size_t site = 0; site < n_sites; ++site) {
    // Project delta_site onto the null space
    arma::vec c = Z.row(site).t();
    const double norm = arma::norm(c);
    if (norm < 1e-15) {
      continue;
    }
    c /= norm;

    // Run annealed IRLS from this seed
    auto [c_result, _] = irls_annealed(Z, c, p);

    arma::vec psi = Z * c_result;
    psi = clean_and_normalize(psi, p.cleanup_tol);

    // Deduplicate by support set
    arma::uvec support_idx = arma::find(arma::abs(psi) > p.cleanup_tol);
    if (support_idx.is_empty()) {
      continue;
    }

    std::vector<arma::uword> support_key(support_idx.begin(), support_idx.end());
    if (seen_supports.insert(support_key).second) {
      cls_list.push_back(psi);
    }
  }

  return cls_list;
}

/// Run the full CLS-rank diagnostic on a Hamiltonian.
///
/// Computes the null space, finds all CLS by site-sweep, and reports
/// rank/dimension/gap diagnostics. An incompleteness_gap > 0 indicates
/// that the CLS set does not span the full flat-band subspace.
inline DiagnosticResult cls_rank_diagnostic(const arma::mat& H, double E_target,
                                            const IrlsParams& p = {}) {
  const arma::mat Z = compute_nullspace(H, E_target, p.svd_tol);
  const size_t nullspace_dim = Z.n_cols;

  std::vector<arma::vec> cls_list = find_all_cls(Z, p);

  if (cls_list.empty()) {
    return {nullspace_dim, 0, 0, nullspace_dim, {}};
  }

  // Project CLS vectors into null-space coordinates and compute rank
  arma::mat C(Z.n_rows, cls_list.size());
  for (size_t i = 0; i < cls_list.size(); ++i) {
    C.col(i) = cls_list[i];
  }
  const arma::mat A = Z.t() * C;
  const size_t cls_rank = arma::rank(A, 1e-10);

  return {nullspace_dim, cls_list.size(), cls_rank,
          nullspace_dim - cls_rank, std::move(cls_list)};
}

}  // namespace l1_cls
