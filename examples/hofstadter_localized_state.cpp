#include <armadillo>
#include <cstddef>
#include <cstdio>

#include "algebra/fermion/basis.h"
#include "algebra/matrix_elements.h"
#include "algebra/model/hofstadter_tight_binding.h"
#include "algorithms/l1_cls.h"

int main() {
  const size_t Nx = 6;
  const size_t Ny = 6;
  const double t = 1.0;
  const double alpha = 1.0 / 3.0;  // 3 magnetic subbands

  HofstadterTightBindingModel model(t, Nx, Ny, alpha);

  // Build single-particle Hamiltonian via many-body machinery
  auto basis = FermionBasis::with_fixed_particle_number_and_spin(model.num_sites, 1, 1);
  arma::cx_mat H = compute_matrix_elements<arma::cx_mat>(basis, model.hamiltonian());

  auto [p, q] = model.flux_fraction();
  const size_t band_size = model.subband_size();

  std::printf("Hofstadter localized state via L1 minimization\n");
  std::printf("  Nx=%zu  Ny=%zu  alpha=%zu/%zu  t=%.2f\n", Nx, Ny, p, q, t);
  std::printf("  Hilbert space dim = %zu\n", model.num_sites);
  std::printf("  Subband size      = %zu\n", band_size);
  std::printf("  Number of bands   = %zu\n\n", model.n_subbands());

  // Extract the lowest magnetic subband
  auto [evals, band_evals, Z] = l1_cls::extract_subband(H, band_size, 0);

  double bandwidth = band_evals(band_evals.n_elem - 1) - band_evals(0);
  double gap = evals(band_size) - band_evals(band_evals.n_elem - 1);
  std::printf("  Lowest band: [%.6f, %.6f]  bandwidth=%.2e  gap=%.6f\n\n", band_evals(0),
              band_evals(band_evals.n_elem - 1), bandwidth, gap);

  // Run IRLS to find the most localized state in this subband
  l1_cls::IrlsParams params;
  auto cls_list = l1_cls::find_all_cls(Z, params);

  std::printf("Found %zu distinct localized states\n\n", cls_list.size());

  // Show diagnostics for the first few
  const size_t n_show = std::min<size_t>(3, cls_list.size());
  for (size_t i = 0; i < n_show; ++i) {
    const auto& psi = cls_list[i];
    arma::uvec support = arma::find(arma::abs(psi) > params.cleanup_tol);

    // Inverse participation ratio
    arma::vec density = arma::square(arma::abs(psi));
    double ipr = arma::dot(density, density);
    double l1 = arma::accu(arma::abs(psi));

    std::printf("State %zu:  support=%llu  L1=%.4f  IPR=%.6f\n", i, support.n_elem, l1, ipr);

    // Check it lives in the subband
    arma::cx_vec projected = Z * (Z.t() * psi);
    double proj_err = arma::norm(psi - projected);
    std::printf("  projection error = %.2e\n\n", proj_err);
  }

  return 0;
}
