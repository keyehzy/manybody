// Block Renormalization Group (BRG) for the attractive Hubbard model on a 2D square lattice.
//
// Tiles the lattice with 2x2 blocks, retains 4 states per block (one from each of the
// (N=0,Sz=0), (N=1,Sz=+1/2), (N=1,Sz=-1/2), (N=2,Sz=0) sectors), and projects the
// Hamiltonian onto this subspace to obtain renormalized parameters (t', U', mu', K').
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

  cxxopts::Options cli("hubbard_block_rg_2d",
                       "Block RG for attractive 2D Hubbard model (2x2 blocks)");
  // clang-format off
  cli.add_options()
      ("t,hopping",     "Hopping amplitude",                cxxopts::value(o.t)->default_value("1.0"))
      ("U,interaction",  "On-site interaction (U<0 attractive)", cxxopts::value(o.U)->default_value("-4.0"))
      ("m,mu",           "Chemical potential",               cxxopts::value(o.mu)->default_value("0.0"))
      ("n,iterations",   "Max RG iterations",                cxxopts::value(o.iterations)->default_value("20"))
      ("mode",           "RG mode: A (density flows) or B (fixed quarter filling)",
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
  const auto geometry = brg::block_2d_2x2();

  const Expression H = brg::build_hubbard_block_hamiltonian(geometry, t, U, mu);

  // Build sector bases using 4 orbitals (the 2x2 block)
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
  double lambda_sum = 0.0;
  int lambda_count = 0;
  double max_spin_diff = 0.0;
  double max_site_diff = 0.0;
  double max_closure_error = 0.0;

  const size_t num_border = geometry.border_sites.size();
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

      // Closure check: <Psi_updown | c^dag_{site,sigma} | Psi_{-sigma}>
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
  for (size_t spi = 0; spi < 2; ++spi) {
    double diff = std::abs(lambdas[0][spi] - lambdas[1][spi]);
    max_site_diff = std::max(max_site_diff, diff);
  }

  // Closure error
  for (size_t si = 0; si < num_border; ++si) {
    for (size_t spi = 0; spi < 2; ++spi) {
      double err = std::abs(lambdas[si][spi] - closures[si][spi]);
      max_closure_error = std::max(max_closure_error, err);
    }
  }

  // ---------------------------------------------------------------------------
  // Pairing correlations (superconductivity diagnostics)
  // ---------------------------------------------------------------------------

  // 1. Pairing amplitude: lambda_pair_i = <psi_N2 | Delta^dag_i | psi_N0>
  //    where Delta^dag_i = c^dag_{i,up} c^dag_{i,down}
  std::vector<double> lambda_pairs(geometry.num_sites);
  double lambda_pair_sum = 0.0;

  for (size_t site = 0; site < geometry.num_sites; ++site) {
    Expression Delta_dag = brg::pair_creation(site);
    arma::cx_mat M_pair =
        compute_rectangular_matrix_elements<arma::cx_mat>(basis_N2, basis_N0, Delta_dag);
    std::complex<double> pair_val = arma::cdot(psi_ud, M_pair * psi_0);
    lambda_pairs[site] = std::abs(pair_val);
    lambda_pair_sum += std::abs(pair_val);
  }

  const double lambda_pair_avg = lambda_pair_sum / geometry.num_sites;

  // Site-independence check for pairing
  double max_pair_site_diff = 0.0;
  for (size_t i = 0; i < geometry.num_sites; ++i) {
    for (size_t j = i + 1; j < geometry.num_sites; ++j) {
      double diff = std::abs(lambda_pairs[i] - lambda_pairs[j]);
      max_pair_site_diff = std::max(max_pair_site_diff, diff);
    }
  }

  // 2. Inter-site pair correlation: <psi_N2 | Delta^dag_i Delta_j | psi_N2> for i != j
  //    This measures pair coherence/delocalization in the two-particle ground state
  double pair_corr_sum = 0.0;
  int pair_corr_count = 0;

  for (size_t i = 0; i < geometry.num_sites; ++i) {
    for (size_t j = 0; j < geometry.num_sites; ++j) {
      if (i == j) continue;

      // Delta^dag_i Delta_j = c^dag_{i,up} c^dag_{i,down} c_{j,down} c_{j,up}
      Expression Delta_dag_i = brg::pair_creation(i);
      Expression Delta_j = brg::pair_annihilation(j);
      Expression pair_hop = Delta_dag_i * Delta_j;

      arma::cx_mat M_corr =
          compute_rectangular_matrix_elements<arma::cx_mat>(basis_N2, basis_N2, pair_hop);
      std::complex<double> corr_val = arma::cdot(psi_ud, M_corr * psi_ud);

      pair_corr_sum += std::real(corr_val);
      ++pair_corr_count;
    }
  }

  const double pair_correlation_avg = (pair_corr_count > 0) ? pair_corr_sum / pair_corr_count : 0.0;

  // Hopping renormalization: t' = nu * lambda^2 * t
  const double t_prime = geometry.nu * lambda_avg * lambda_avg * t;

  return brg::make_zero_t_result(t_prime, U_prime, mu_prime, K_prime, E1, E2, E3, E4, lambda_avg,
                                 max_spin_diff, max_site_diff, max_closure_error, lambda_pair_avg,
                                 max_pair_site_diff, pair_correlation_avg);
}

// ---------------------------------------------------------------------------
// Mode B: tune mu for quarter filling (N=1 per block)
// ---------------------------------------------------------------------------

double tune_mu_for_quarter_filling(double t, double U, bool& window_exists) {
  const auto geometry = brg::block_2d_2x2();

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

  std::cerr << "=== Block RG for 2D Hubbard (2x2 blocks) ===" << std::endl;
  std::cerr << "Mode: " << (mode_b ? "B (fixed quarter filling)" : "A (density flows)")
            << std::endl;
  std::cerr << "Initial: t=" << opts.t << ", U=" << opts.U << ", mu=" << opts.mu << std::endl;
  std::cerr << std::endl;

  // Print header
  std::cout << std::setprecision(8) << std::fixed;
  std::cout << "# iter      t             U             mu            U/t           mu/t"
            << "          lambda        spin_diff     site_diff     closure_err"
            << "   lambda_pair   pair_corr";
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
      mu = tune_mu_for_quarter_filling(t, U, window_exists);
    }

    brg::BrgStepResult result = brg_step(t, U, mu);

    // Output row
    std::cout << std::setw(4) << n << "  " << std::setw(13) << t << " " << std::setw(13) << U << " "
              << std::setw(13) << mu << " " << std::setw(13) << (U / t) << " " << std::setw(13)
              << (mu / t) << " " << std::setw(13) << result.lambda_avg << " " << std::setw(13)
              << result.lambda_spin_diff << " " << std::setw(13) << result.lambda_site_diff << " "
              << std::setw(13) << result.closure_error << " " << std::setw(13)
              << result.lambda_pair_avg << " " << std::setw(13) << result.pair_correlation_avg;
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
    std::cerr << "  lambda_pair=" << result.lambda_pair_avg
              << " pair_corr=" << result.pair_correlation_avg
              << " pair_site_diff=" << result.lambda_pair_site_diff << std::endl;
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
