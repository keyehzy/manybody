#pragma once

#include <armadillo>
#include <cmath>
#include <complex>
#include <set>
#include <type_traits>
#include <vector>

/// L1-norm minimization for extracting localized states from degenerate
/// subspaces.
///
/// Given an orthonormal basis Z for a target subspace (e.g. a flat-band
/// null space or a magnetic subband), finds the most localized state by
/// minimizing the L1-norm of psi = Z * c over coefficient vectors c,
/// using Iteratively Reweighted Least Squares (IRLS) with geometric
/// eps-annealing.
///
/// All core routines are templated on the scalar type T (double or
/// cx_double) so the same algorithm handles both real flat-band models
/// and complex Hofstadter / Landau-level subspaces.
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

template <typename T>
struct DiagnosticResult {
  size_t nullspace_dim;
  size_t n_cls_found;
  size_t cls_rank;
  size_t incompleteness_gap;
  std::vector<arma::Col<T>> cls_list;
};

// ── helpers ─────────────────────────────────────────────────────────

namespace detail {

/// Fix the global phase ambiguity of `candidate` using overlap with
/// `reference`.  For real vectors this is a sign flip; for complex
/// vectors it is a full U(1) rotation.
template <typename T>
inline arma::Col<T> align_phase(const arma::Col<T>& candidate,
                                const arma::Col<T>& reference) {
  if constexpr (std::is_same_v<T, double>) {
    if (arma::dot(reference, candidate) < 0.0) {
      return -candidate;
    }
    return candidate;
  } else {
    const std::complex<double> overlap = arma::cdot(reference, candidate);
    if (std::abs(overlap) < 1e-15) {
      return candidate;
    }
    return candidate * std::exp(std::complex<double>(0.0, -std::arg(overlap)));
  }
}

/// Phase convention: make the largest-magnitude component real and
/// positive.
template <typename T>
inline arma::Col<T> fix_phase_convention(arma::Col<T>& psi) {
  const arma::uword idx = arma::index_max(arma::abs(psi));
  if constexpr (std::is_same_v<T, double>) {
    if (psi(idx) < 0.0) {
      psi = -psi;
    }
  } else {
    const double phase = std::arg(psi(idx));
    psi *= std::exp(std::complex<double>(0.0, -phase));
  }
  return psi;
}

}  // namespace detail

// ── core routines ───────────────────────────────────────────────────

/// Phase-convention cleanup: make the largest-magnitude component
/// positive (real), threshold numerical noise, and re-normalize.
template <typename T>
inline arma::Col<T> clean_and_normalize(arma::Col<T> psi, double cleanup_tol) {
  double norm = arma::norm(psi);
  if (norm < 1e-15) {
    return psi;
  }
  psi /= norm;

  detail::fix_phase_convention(psi);

  // Zero out noise
  psi.elem(arma::find(arma::abs(psi) < cleanup_tol)).zeros();

  // Re-normalize
  norm = arma::norm(psi);
  if (norm > 0.0) {
    psi /= norm;
  }
  return psi;
}

/// Extract an orthonormal basis for the null space of (H - E_target * I).
///
/// Returns Z whose columns span ker(H - E_target * I), identified via SVD
/// as the right singular vectors whose singular values fall below svd_tol.
template <typename T>
inline arma::Mat<T> compute_nullspace(const arma::Mat<T>& H, double E_target,
                                      double svd_tol = 1e-10) {
  const size_t dim = H.n_rows;
  arma::Mat<T> M = H;
  for (size_t i = 0; i < dim; ++i) {
    M(i, i) -= T(E_target);
  }

  arma::Mat<T> U;
  arma::vec s;
  arma::Mat<T> V;
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
  arma::Mat<T> Z = V.cols(V.n_cols - null_count, V.n_cols - 1);
  return Z;
}

/// Extract an orthonormal basis for one magnetic subband.
///
/// Diagonalizes H, groups eigenstates into q subbands of equal size
/// (where alpha = p/q), and returns the eigenvectors spanning the
/// selected band_index.
///
/// Returns {all_eigenvalues, band_eigenvalues, Z}.
template <typename T>
inline std::tuple<arma::vec, arma::vec, arma::Mat<T>>
extract_subband(const arma::Mat<T>& H, size_t band_size,
                size_t band_index = 0) {
  arma::vec evals;
  arma::Mat<T> evecs;
  arma::eig_sym(evals, evecs, H);

  const size_t n_bands = H.n_rows / band_size;
  if (band_index >= n_bands) {
    throw std::out_of_range("band_index out of range");
  }

  const size_t start = band_index * band_size;
  const size_t stop = start + band_size;

  arma::vec band_evals = evals.subvec(start, stop - 1);
  arma::Mat<T> Z = evecs.cols(start, stop - 1);

  return {evals, band_evals, Z};
}

/// Core IRLS loop with geometric eps-annealing.
///
/// Finds the coefficient vector c that minimizes ||Z * c||_1 subject to
/// ||c|| = 1, by iteratively solving a weighted eigenvalue problem.
/// Returns {c, total_iterations}.
template <typename T>
inline std::pair<arma::Col<T>, size_t> irls_annealed(const arma::Mat<T>& Z,
                                                     arma::Col<T> c,
                                                     const IrlsParams& p) {
  const arma::vec eps_schedule = arma::logspace(
      std::log10(p.eps_start), std::log10(p.eps_end), p.n_eps_stages);
  size_t total_iter = 0;

  for (size_t stage = 0; stage < p.n_eps_stages; ++stage) {
    const double eps = eps_schedule(stage);
    const double eps_sq = eps * eps;

    for (size_t iter = 0; iter < p.max_iter_per_stage; ++iter) {
      const arma::Col<T> psi = Z * c;

      // Reweighting: w_i = 1 / sqrt(|psi_i|^2 + eps^2)
      const arma::vec w =
          1.0 / arma::sqrt(arma::square(arma::abs(psi)) + eps_sq);

      // Weighted Gram matrix: G = Z^H diag(w) Z  (Hermitian)
      arma::Mat<T> G = Z.t() * arma::diagmat(w) * Z;
      G = 0.5 * (G + G.t());  // enforce exact Hermiticity

      // New c = eigenvector of G with smallest eigenvalue
      arma::vec eigval;
      arma::Mat<T> eigvec;
      arma::eig_sym(eigval, eigvec, G);
      arma::Col<T> c_new = eigvec.col(0);

      // Phase alignment with previous c
      c_new = detail::align_phase(c_new, c);

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

/// Find all unique localized states by seeding IRLS from every site.
///
/// For each site index, projects a delta vector onto the subspace to
/// obtain a biased starting vector, then runs annealed IRLS.  Results
/// are deduplicated by their support set (the set of site indices with
/// nonzero amplitude).
template <typename T>
inline std::vector<arma::Col<T>> find_all_cls(const arma::Mat<T>& Z,
                                              const IrlsParams& p) {
  const size_t n_sites = Z.n_rows;

  std::set<std::vector<arma::uword>> seen_supports;
  std::vector<arma::Col<T>> cls_list;

  for (size_t site = 0; site < n_sites; ++site) {
    // Project delta_site onto the subspace
    arma::Col<T> c = Z.row(site).t();
    const double norm = arma::norm(c);
    if (norm < 1e-15) {
      continue;
    }
    c /= norm;

    // Run annealed IRLS from this seed
    auto [c_result, _] = irls_annealed(Z, c, p);

    arma::Col<T> psi = Z * c_result;
    psi = clean_and_normalize(psi, p.cleanup_tol);

    // Deduplicate by support set
    arma::uvec support_idx = arma::find(arma::abs(psi) > p.cleanup_tol);
    if (support_idx.is_empty()) {
      continue;
    }

    std::vector<arma::uword> support_key(support_idx.begin(),
                                         support_idx.end());
    if (seen_supports.insert(support_key).second) {
      cls_list.push_back(psi);
    }
  }

  return cls_list;
}

/// Run the full CLS-rank diagnostic on a Hamiltonian.
///
/// Computes the null space, finds all CLS by site-sweep, and reports
/// rank/dimension/gap diagnostics.  An incompleteness_gap > 0 indicates
/// that the CLS set does not span the full flat-band subspace.
template <typename T>
inline DiagnosticResult<T> cls_rank_diagnostic(const arma::Mat<T>& H,
                                               double E_target,
                                               const IrlsParams& p = {}) {
  const arma::Mat<T> Z = compute_nullspace(H, E_target, p.svd_tol);
  const size_t nullspace_dim = Z.n_cols;

  std::vector<arma::Col<T>> cls_list = find_all_cls(Z, p);

  if (cls_list.empty()) {
    return {nullspace_dim, 0, 0, nullspace_dim, {}};
  }

  // Project CLS vectors into null-space coordinates and compute rank
  arma::Mat<T> C(Z.n_rows, cls_list.size());
  for (size_t i = 0; i < cls_list.size(); ++i) {
    C.col(i) = cls_list[i];
  }
  const arma::Mat<T> A = Z.t() * C;
  const size_t cls_rank = arma::rank(A, 1e-10);

  return {nullspace_dim, cls_list.size(), cls_rank, nullspace_dim - cls_rank,
          std::move(cls_list)};
}

}  // namespace l1_cls
