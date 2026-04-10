#include <cstddef>
#include <cstdio>

#include <armadillo>

#include "algebra/fermion/basis.h"
#include "algebra/matrix_elements.h"
#include "algebra/model/sawtooth_tight_binding.h"
#include "algorithms/l1_cls.h"

int main() {
  const size_t num_cells = 8;
  const double t_base = 1.0;
  const double t_apex = std::sqrt(2.0) * t_base;

  // Build model and single-particle Hamiltonian via the many-body machinery
  SawtoothTightBindingModel model(t_base, t_apex, num_cells);
  // Spin-polarized single-particle basis (only Spin::Up orbitals)
  auto basis = FermionBasis::with_fixed_particle_number_and_spin(model.num_sites, 1, 1);
  auto H_cx = compute_matrix_elements<arma::cx_mat>(basis, model.hamiltonian());
  arma::mat H = arma::real(H_cx);

  const double E_fb = model.flat_band_energy();

  std::printf("Sawtooth CLS via L1 minimization\n");
  std::printf("  num_cells = %zu, t_base = %.4f, t_apex = %.6f\n",
              num_cells, t_base, t_apex);
  std::printf("  Hilbert space dim = %zu\n", basis.set.size());
  std::printf("  Flat-band energy  = %.10f\n\n", E_fb);

  // Run CLS diagnostic
  l1_cls::IrlsParams params;
  auto result = l1_cls::cls_rank_diagnostic<double>(H, E_fb, params);

  std::printf("Diagnostic:\n");
  std::printf("  Nullspace dim       = %zu\n", result.nullspace_dim);
  std::printf("  CLS found           = %zu\n", result.n_cls_found);
  std::printf("  CLS rank            = %zu\n", result.cls_rank);
  std::printf("  Incompleteness gap  = %zu\n\n", result.incompleteness_gap);

  // Print each CLS
  for (size_t i = 0; i < result.cls_list.size(); ++i) {
    const auto& psi = result.cls_list[i];
    arma::uvec support = arma::find(arma::abs(psi) > params.cleanup_tol);

    std::printf("CLS %zu (support = %llu sites):\n", i, support.n_elem);
    for (size_t j = 0; j < support.n_elem; ++j) {
      const arma::uword site = support(j);
      const char* sublattice = (site % 2 == 0) ? "base" : "apex";
      const size_t cell = site / 2;
      std::printf("    site %3llu (%s_%zu): %+.10f\n",
                  static_cast<unsigned long long>(site), sublattice, cell, psi(site));
    }

    // Verify eigenstate
    const double residual = arma::norm(H * psi - E_fb * psi);
    std::printf("    eigenstate residual = %.2e\n\n", residual);
  }

  return 0;
}
