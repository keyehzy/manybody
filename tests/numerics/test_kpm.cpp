#include <armadillo>
#include <catch2/catch.hpp>
#include <cmath>
#include <numbers>

#include "numerics/kpm.h"
#include "numerics/rescaling.h"

namespace test_kpm {

TEST_CASE("kpm_jackson_kernel_properties") {
  const size_t N = 50;
  auto g = kpm::jackson_kernel(N);

  // Jackson kernel should have g_0 = 1
  CHECK(std::abs(g[0] - 1.0) < 1e-10);

  // All damping factors should be non-negative and <= 1
  for (size_t n = 0; n <= N; ++n) {
    CHECK(g[n] >= -1e-10);
    CHECK(g[n] <= 1.0 + 1e-10);
  }

  // Damping factors should decrease monotonically
  for (size_t n = 1; n <= N; ++n) {
    CHECK(g[n] <= g[n - 1] + 1e-10);
  }
}

TEST_CASE("kpm_lorentz_kernel_properties") {
  const size_t N = 50;
  auto g = kpm::lorentz_kernel(N);

  // Lorentz kernel should have g_0 = 1
  CHECK(std::abs(g[0] - 1.0) < 1e-10);

  // All damping factors should be non-negative and <= 1
  for (size_t n = 0; n <= N; ++n) {
    CHECK(g[n] >= -1e-10);
    CHECK(g[n] <= 1.0 + 1e-10);
  }
}

TEST_CASE("kpm_projector_moments_sum_rule") {
  const size_t M = 100;

  // For epsilon = 0 (half the spectrum), mu_0 should be 0.5
  auto mu_half = kpm::projector_moments(M, 0.0);
  CHECK(std::abs(mu_half[0] - 0.5) < 1e-10);

  // For epsilon = -1 (nothing below Fermi energy), mu_0 should be 0
  auto mu_empty = kpm::projector_moments(M, -1.0);
  CHECK(std::abs(mu_empty[0]) < 1e-10);

  // For epsilon = 1 (entire spectrum below Fermi energy), mu_0 should be 1
  auto mu_full = kpm::projector_moments(M, 1.0);
  CHECK(std::abs(mu_full[0] - 1.0) < 1e-10);
}

TEST_CASE("kpm_rescaling_preserves_spectrum") {
  arma::mat H(4, 4, arma::fill::zeros);
  H.diag() = arma::vec{-2.0, -1.0, 1.0, 3.0};

  auto r = rescaling::estimate_rescaling(H);
  arma::mat H_scaled = rescaling::rescale_hamiltonian(H, r);

  arma::vec eigvals = arma::eig_sym(H_scaled);

  // All eigenvalues should be in [-1, 1]
  CHECK(eigvals.min() >= -1.0 - 1e-10);
  CHECK(eigvals.max() <= 1.0 + 1e-10);
}

TEST_CASE("kpm_projector_reproduces_exact_projector") {
  // Simple 4x4 symmetric matrix with known eigenvalues
  arma::mat H(4, 4, arma::fill::zeros);
  H.diag() = arma::vec{-2.0, -0.5, 0.5, 2.0};
  H(0, 1) = 0.1;
  H(1, 0) = 0.1;
  H(2, 3) = 0.1;
  H(3, 2) = 0.1;

  // Exact projector onto states below E_fermi = 0
  arma::vec eigvals;
  arma::mat eigvecs;
  arma::eig_sym(eigvals, eigvecs, H);

  arma::mat P_exact(4, 4, arma::fill::zeros);
  for (size_t i = 0; i < 4; ++i) {
    if (eigvals(i) < 0.0) {
      P_exact += eigvecs.col(i) * eigvecs.col(i).t();
    }
  }

  // KPM projector with high order should be accurate
  kpm::KPMProjector kpm_proj(H, 200, 0.0);
  arma::mat P_kpm = kpm_proj.build_matrix();

  double error = arma::norm(P_kpm - P_exact, "fro");
  CHECK(error < 1e-2);
}

TEST_CASE("kpm_projector_apply_matches_matrix") {
  arma::mat H(6, 6, arma::fill::zeros);
  H.diag().fill(0.0);
  for (size_t i = 0; i + 1 < 6; ++i) {
    H(i, i + 1) = -1.0;
    H(i + 1, i) = -1.0;
  }

  kpm::KPMProjector proj(H, 100, 0.0);
  arma::mat P = proj.build_matrix();

  // Apply to a random vector
  arma::vec v = arma::randn<arma::vec>(6);
  arma::vec Pv_matrix = P * v;
  arma::vec Pv_apply = proj.apply(v);

  double error = arma::norm(Pv_matrix - Pv_apply);
  CHECK(error < 1e-10);
}

TEST_CASE("kpm_projector_idempotent") {
  arma::mat H(4, 4, arma::fill::zeros);
  H.diag() = arma::vec{-1.5, -0.5, 0.5, 1.5};

  kpm::KPMProjector proj(H, 200, 0.0);
  arma::mat P = proj.build_matrix();

  // A projector satisfies P^2 = P
  arma::mat P_squared = P * P;
  double error = arma::norm(P_squared - P, "fro");
  CHECK(error < 0.1);  // KPM approximation has some error
}

TEST_CASE("kpm_projector_trace_counts_states") {
  arma::mat H(6, 6, arma::fill::zeros);
  H.diag() = arma::vec{-2.0, -1.0, -0.5, 0.5, 1.0, 2.0};

  // E_fermi = 0 should give 3 occupied states
  kpm::KPMProjector proj(H, 200, 0.0);
  arma::mat P = proj.build_matrix();

  double trace = arma::trace(P);
  CHECK(std::abs(trace - 3.0) < 0.1);
}

TEST_CASE("kpm_projector_implements_linear_operator") {
  arma::mat H(4, 4, arma::fill::zeros);
  H.diag() = arma::vec{-1.5, -0.5, 0.5, 1.5};

  kpm::KPMProjector proj(H, 50, 0.0);

  // Check dimension
  CHECK(proj.dimension() == 4);

  // Check that it can be used through base class pointer
  LinearOperator<arma::vec>* op = &proj;
  CHECK(op->dimension() == 4);

  arma::vec v = arma::randn<arma::vec>(4);
  arma::vec result_direct = proj.apply(v);
  arma::vec result_via_base = op->apply(v);

  CHECK(arma::norm(result_direct - result_via_base) < 1e-14);
}

TEST_CASE("sparse_kpm_projector_implements_linear_operator") {
  arma::sp_mat H(4, 4);
  H.diag() = arma::vec{-1.5, -0.5, 0.5, 1.5};

  kpm::SparseKPMProjector proj(H, 50, -2.0, 2.0, 0.0);

  // Check dimension
  CHECK(proj.dimension() == 4);

  // Check that it can be used through base class pointer
  LinearOperator<arma::vec>* op = &proj;
  CHECK(op->dimension() == 4);

  arma::vec v = arma::randn<arma::vec>(4);
  arma::vec result_direct = proj.apply(v);
  arma::vec result_via_base = op->apply(v);

  CHECK(arma::norm(result_direct - result_via_base) < 1e-14);
}

TEST_CASE("sparse_kpm_projector_apply_matches_dense") {
  arma::mat H_dense(4, 4, arma::fill::zeros);
  H_dense.diag() = arma::vec{-1.5, -0.5, 0.5, 1.5};
  arma::sp_mat H_sparse(H_dense);

  kpm::KPMProjector dense_proj(H_dense, 100, 0.0);
  kpm::SparseKPMProjector sparse_proj(H_sparse, 100, -2.0, 2.0, 0.0);

  arma::vec v = arma::randn<arma::vec>(4);
  arma::vec result_dense = dense_proj.apply(v);
  arma::vec result_sparse = sparse_proj.apply(v);

  // Results should be similar (not exact due to different rescaling)
  CHECK(arma::norm(result_dense - result_sparse) < 0.1);
}

}  // namespace test_kpm
