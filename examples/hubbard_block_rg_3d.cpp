// Block Renormalization Group (BRG) for the attractive Hubbard model on a 3D simple cubic lattice.
//
// Tiles the lattice with 2x2x2 blocks (8 sites each), retains 4 states per block (one from each
// of the (N=0,Sz=0), (N=1,Sz=+1/2), (N=1,Sz=-1/2), (N=2,Sz=0) sectors), and projects the
// Hamiltonian onto this subspace to obtain renormalized parameters (t', U', mu', K').
//
// 1/8 filling: 1 electron per block.
//
// Block geometry (2x2x2, site index = 4*z + 2*y + x):
//
//   z=0 layer:        z=1 layer:
//     2---3              6---7
//     |   |              |   |
//     0---1              4---5
//
//   Vertical bonds (z-direction): 0-4, 1-5, 2-6, 3-7
//
// Intra-block bonds (12 total, open boundary):
//   x-direction: (0,1), (2,3), (4,5), (6,7)
//   y-direction: (0,2), (1,3), (4,6), (5,7)
//   z-direction: (0,4), (1,5), (2,6), (3,7)
//
// Inter-block connectivity (4 bonds per face):
//   +x: {1,3,5,7} <-> neighbor's {0,2,4,6}   => nu = 4
//   +y: {2,3,6,7} <-> neighbor's {0,1,4,5}   => nu = 4
//   +z: {4,5,6,7} <-> neighbor's {0,1,2,3}   => nu = 4
//
// Reference: Wang, Kais, Levine, Int. J. Mol. Sci. 2002, 3, 4-16.

#include <armadillo>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>

#include "algebra/fermion/basis.h"
#include "algebra/matrix_elements.h"
#include "algorithms/brg/brg.h"
#include "cxxopts.hpp"

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

struct CliOptions {
  double t = 1.0;
  double U = -4.0;
  double mu = 0.0;
  int iterations = 20;
  std::string mode = "A";
};

CliOptions parse_cli_options(int argc, char** argv) {
  CliOptions o;

  cxxopts::Options cli("hubbard_block_rg_3d",
                       "Block RG for attractive 3D Hubbard model (2x2x2 blocks)");
  // clang-format off
  cli.add_options()
      ("t,hopping",     "Hopping amplitude",                cxxopts::value(o.t)->default_value("1.0"))
      ("U,interaction",  "On-site interaction (U<0 attractive)", cxxopts::value(o.U)->default_value("-4.0"))
      ("m,mu",           "Chemical potential",               cxxopts::value(o.mu)->default_value("0.0"))
      ("n,iterations",   "Max RG iterations",                cxxopts::value(o.iterations)->default_value("20"))
      ("mode",           "RG mode: A (density flows) or B (fixed 1/8 filling)",
                                                             cxxopts::value(o.mode)->default_value("A"))
      ("h,help",         "Print usage");
  // clang-format on

  try {
    auto result = cli.parse(argc, argv);
    if (result.count("help")) {
      std::cout << cli.help() << "\n";
      std::exit(0);
    }
  } catch (const std::exception& e) {
    std::cerr << "Argument error: " << e.what() << "\n" << cli.help() << "\n";
    std::exit(1);
  }

  return o;
}

// ---------------------------------------------------------------------------
// Single BRG step
// ---------------------------------------------------------------------------

brg::BrgStepResult brg_step(double t, double U, double mu) {
  const auto geometry = brg::block_3d_2x2x2();

  const Expression H = brg::build_hubbard_block_hamiltonian(geometry, t, U, mu);

  // Build sector bases using 8 orbitals (the 2x2x2 block)
  Basis basis_N0 = Basis::with_fixed_particle_number_and_spin(geometry.num_sites, 0, 0);
  Basis basis_N1_up = Basis::with_fixed_particle_number_and_spin(geometry.num_sites, 1, 1);
  Basis basis_N1_down = Basis::with_fixed_particle_number_and_spin(geometry.num_sites, 1, -1);
  Basis basis_N2 = Basis::with_fixed_particle_number_and_spin(geometry.num_sites, 2, 0);

  // Diagonalize each sector
  brg::SectorResult res_N0 = brg::diagonalize_sector(basis_N0, H);
  brg::SectorResult res_N1_up = brg::diagonalize_sector(basis_N1_up, H);
  brg::SectorResult res_N1_down = brg::diagonalize_sector(basis_N1_down, H);
  brg::SectorResult res_N2 = brg::diagonalize_sector(basis_N2, H);

  const double E1 = res_N0.eigenvalues(0);
  const double E2 = res_N2.eigenvalues(0);
  const double E3 = res_N1_up.eigenvalues(0);
  const double E4 = res_N1_down.eigenvalues(0);

  // Ground eigenvectors
  const arma::cx_vec& psi_0 = res_N0.eigenvectors.col(0);
  const arma::cx_vec& psi_up = res_N1_up.eigenvectors.col(0);
  const arma::cx_vec& psi_down = res_N1_down.eigenvectors.col(0);
  const arma::cx_vec& psi_ud = res_N2.eigenvectors.col(0);

  // Renormalized onsite parameters (Eqs. 14-16)
  const double U_prime = E1 + E2 - 2.0 * E3;
  const double mu_prime = E1 - E3;
  const double K_prime = E1;

  // Compute lambda: <Psi_sigma | c^dag_{i,sigma} | Psi_0>
  const size_t num_border = geometry.border_sites.size();

  double lambda_sum = 0.0;
  int lambda_count = 0;
  double max_spin_diff = 0.0;
  double max_site_diff = 0.0;
  double max_closure_error = 0.0;

  std::vector<std::array<double, 2>> lambdas(num_border);
  std::vector<std::array<double, 2>> closures(num_border);

  for (size_t si = 0; si < num_border; ++si) {
    const size_t site = geometry.border_sites[si];

    for (size_t spi = 0; spi < 2; ++spi) {
      auto sigma = (spi == 0) ? Operator::Spin::Up : Operator::Spin::Down;

      Expression c_dag(creation(sigma, site));

      const Basis& row_basis = (spi == 0) ? basis_N1_up : basis_N1_down;
      const arma::cx_vec& psi_sigma = (spi == 0) ? psi_up : psi_down;

      arma::cx_mat M =
          compute_rectangular_matrix_elements<arma::cx_mat>(row_basis, basis_N0, c_dag);
      std::complex<double> lambda_val = arma::cdot(psi_sigma, M * psi_0);

      lambdas[si][spi] = std::abs(lambda_val);
      lambda_sum += std::abs(lambda_val);
      ++lambda_count;

      // Closure check
      const Basis& closure_col_basis = (spi == 0) ? basis_N1_down : basis_N1_up;
      const arma::cx_vec& psi_minus_sigma = (spi == 0) ? psi_down : psi_up;

      arma::cx_mat M_closure =
          compute_rectangular_matrix_elements<arma::cx_mat>(basis_N2, closure_col_basis, c_dag);
      std::complex<double> closure_val = arma::cdot(psi_ud, M_closure * psi_minus_sigma);

      closures[si][spi] = std::abs(closure_val);
    }
  }

  const double lambda_avg = lambda_sum / lambda_count;

  // Spin-independence check
  for (size_t si = 0; si < num_border; ++si) {
    double diff = std::abs(lambdas[si][0] - lambdas[si][1]);
    max_spin_diff = std::max(max_spin_diff, diff);
  }

  // Site-independence check
  for (size_t si = 0; si < num_border; ++si) {
    for (size_t sj = si + 1; sj < num_border; ++sj) {
      for (size_t spi = 0; spi < 2; ++spi) {
        double diff = std::abs(lambdas[si][spi] - lambdas[sj][spi]);
        max_site_diff = std::max(max_site_diff, diff);
      }
    }
  }

  // Closure error
  for (size_t si = 0; si < num_border; ++si) {
    for (size_t spi = 0; spi < 2; ++spi) {
      double err = std::abs(lambdas[si][spi] - closures[si][spi]);
      max_closure_error = std::max(max_closure_error, err);
    }
  }

  // Hopping renormalization: t' = nu * lambda^2 * t
  const double t_prime = geometry.nu * lambda_avg * lambda_avg * t;

  return brg::make_zero_t_result(t_prime, U_prime, mu_prime, K_prime, E1, E2, E3, E4, lambda_avg,
                                 max_spin_diff, max_site_diff, max_closure_error);
}

// ---------------------------------------------------------------------------
// Mode B: tune mu for 1/8 filling (N=1 per block)
// ---------------------------------------------------------------------------

double tune_mu_for_eighth_filling(double t, double U, bool& window_exists) {
  const auto geometry = brg::block_3d_2x2x2();

  const Expression H0 = brg::build_hubbard_block_hamiltonian(geometry, t, U, 0.0);

  Basis basis_N0 = Basis::with_fixed_particle_number_and_spin(geometry.num_sites, 0, 0);
  Basis basis_N1_up = Basis::with_fixed_particle_number_and_spin(geometry.num_sites, 1, 1);
  Basis basis_N1_down = Basis::with_fixed_particle_number_and_spin(geometry.num_sites, 1, -1);
  Basis basis_N2 = Basis::with_fixed_particle_number_and_spin(geometry.num_sites, 2, 0);

  brg::SectorResult res_N0 = brg::diagonalize_sector(basis_N0, H0);
  brg::SectorResult res_N1_up = brg::diagonalize_sector(basis_N1_up, H0);
  brg::SectorResult res_N1_down = brg::diagonalize_sector(basis_N1_down, H0);
  brg::SectorResult res_N2 = brg::diagonalize_sector(basis_N2, H0);

  const double e0 = res_N0.eigenvalues(0);
  const double e1 = std::min(res_N1_up.eigenvalues(0), res_N1_down.eigenvalues(0));
  const double e2 = res_N2.eigenvalues(0);

  const double mu_low = e1 - e0;
  const double mu_high = e2 - e1;

  if (mu_low < mu_high) {
    window_exists = true;
    return (mu_low + mu_high) / 2.0;
  } else {
    window_exists = false;
    return (mu_low + mu_high) / 2.0;
  }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main(int argc, char** argv) {
  CliOptions opts = parse_cli_options(argc, argv);

  const bool mode_b = (opts.mode == "B" || opts.mode == "b");

  std::cerr << "=== Block RG for 3D Hubbard (2x2x2 blocks) ===" << std::endl;
  std::cerr << "Mode: " << (mode_b ? "B (fixed 1/8 filling)" : "A (density flows)") << std::endl;
  std::cerr << "Initial: t=" << opts.t << ", U=" << opts.U << ", mu=" << opts.mu << std::endl;
  std::cerr << std::endl;

  // Print header
  std::cout << std::setprecision(8) << std::fixed;
  std::cout << "# iter      t             U             mu            U/t           mu/t"
            << "          lambda        spin_diff     site_diff     closure_err";
  if (mode_b) {
    std::cout << "   window";
  }
  std::cout << "\n";

  double t = opts.t;
  double U = opts.U;
  double mu = opts.mu;

  for (int n = 0; n < opts.iterations; ++n) {
    bool window_exists = true;

    // Mode B: tune mu before the RG step
    if (mode_b) {
      mu = tune_mu_for_eighth_filling(t, U, window_exists);
    }

    brg::BrgStepResult result = brg_step(t, U, mu);

    // Output row
    std::cout << std::setw(4) << n << "  " << std::setw(13) << t << " " << std::setw(13) << U << " "
              << std::setw(13) << mu << " " << std::setw(13) << (U / t) << " " << std::setw(13)
              << (mu / t) << " " << std::setw(13) << result.lambda_avg << " " << std::setw(13)
              << result.lambda_spin_diff << " " << std::setw(13) << result.lambda_site_diff << " "
              << std::setw(13) << result.closure_error;
    if (mode_b) {
      std::cout << "   " << (window_exists ? "yes" : "NO");
    }
    std::cout << "\n";

    // Diagnostics to stderr
    std::cerr << "Iteration " << n << ": E1=" << result.E1 << " E2=" << result.E2
              << " E3=" << result.E3 << " E4=" << result.E4 << std::endl;
    std::cerr << "  |E3-E4| = " << std::abs(result.E3 - result.E4) << " (spin degeneracy check)"
              << std::endl;
    std::cerr << "  U'=" << result.U_prime << " mu'=" << result.mu_prime << " K'=" << result.K_prime
              << " t'=" << result.t_prime << std::endl;
    std::cerr << "  lambda=" << result.lambda_avg << " closure_err=" << result.closure_error
              << std::endl;
    std::cerr << std::endl;

    // Convergence/divergence checks
    const double Ut_ratio = std::abs(U / t);
    if (Ut_ratio > 1e6) {
      std::cerr << "Divergence detected: |U/t| > 1e6. Stopping.\n";
      break;
    }
    if (std::abs(result.t_prime) < 1e-15) {
      std::cerr << "t' ~ 0. Stopping.\n";
      break;
    }

    // Check convergence
    const double dt = std::abs(result.t_prime - t);
    const double dU = std::abs(result.U_prime - U);
    const double dmu = std::abs(result.mu_prime - mu);
    if (dt / std::abs(t) < 1e-10 && dU / (std::abs(U) + 1e-15) < 1e-10 &&
        dmu / (std::abs(mu) + 1e-15) < 1e-10) {
      std::cerr << "Converged.\n";
      t = result.t_prime;
      U = result.U_prime;
      mu = mode_b ? mu : result.mu_prime;
      break;
    }

    // Update parameters for next iteration
    t = result.t_prime;
    U = result.U_prime;
    mu = mode_b ? mu : result.mu_prime;
  }

  std::cerr << "Final parameters: t=" << t << ", U=" << U << ", mu=" << mu << ", U/t=" << (U / t)
            << std::endl;

  return 0;
}
