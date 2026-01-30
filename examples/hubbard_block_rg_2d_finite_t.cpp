// Finite-temperature Block Renormalization Group (BRG) for the attractive Hubbard model on a 2D
// square lattice.
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
#include <limits>
#include <string>

#include "algebra/basis.h"
#include "algebra/expression.h"
#include "algebra/matrix_elements.h"
#include "cxxopts.hpp"

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

constexpr double kMinTemperature = 1e-15;
constexpr double kMaxBeta = 1e15;

bool use_zero_temperature(double T) {
  if (T <= 0.0) {
    return true;
  }
  if (T < kMinTemperature) {
    return true;
  }
  const double beta = 1.0 / T;
  return beta > kMaxBeta;
}

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

struct CliOptions {
  double t = 1.0;
  double U = -4.0;
  double mu = 0.0;
  int iterations = 20;
  std::string mode = "A";
  double temperature = 0.0;
  std::string temp_mode = "fixed";  // fixed or rescaled
};

CliOptions parse_cli_options(int argc, char** argv) {
  CliOptions o;

  cxxopts::Options cli("hubbard_block_rg_2d_finite_t",
                       "Finite-T Block RG for attractive 2D Hubbard model (2x2 blocks)");
  // clang-format off
  cli.add_options()
      ("t,hopping",     "Hopping amplitude",                     cxxopts::value(o.t)->default_value("1.0"))
      ("U,interaction", "On-site interaction (U<0 attractive)",   cxxopts::value(o.U)->default_value("-4.0"))
      ("m,mu",          "Chemical potential",                    cxxopts::value(o.mu)->default_value("0.0"))
      ("n,iterations",  "Max RG iterations",                     cxxopts::value(o.iterations)->default_value("20"))
      ("mode",          "RG mode: A (density flows) or B (fixed quarter filling)",
                                                                 cxxopts::value(o.mode)->default_value("A"))
      ("temperature",   "Temperature T (default 0.0)",           cxxopts::value(o.temperature)->default_value("0.0"))
      ("temp-mode",     "Temperature mode: fixed or rescaled",   cxxopts::value(o.temp_mode)->default_value("fixed"))
      ("h,help",        "Print usage");
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

  if (o.temperature < 0.0) {
    std::cerr << "Temperature must be non-negative.\n";
    std::exit(1);
  }

  if (o.temp_mode != "fixed" && o.temp_mode != "rescaled") {
    std::cerr << "temp-mode must be 'fixed' or 'rescaled'.\n";
    std::exit(1);
  }

  return o;
}

// ---------------------------------------------------------------------------
// Block Hamiltonian (2x2 open-BC Hubbard)
// ---------------------------------------------------------------------------

// Sites:  0:(0,0)  1:(0,1)
//         2:(1,0)  3:(1,1)
// Bonds:  (0-1), (0-2), (1-3), (2-3)

Expression build_block_hamiltonian(double t, double U, double mu) {
  constexpr size_t num_sites = 4;

  // Intra-block bonds (open boundary on the 2x2 grid)
  const std::pair<size_t, size_t> bonds[] = {{0, 1}, {0, 2}, {1, 3}, {2, 3}};

  Expression H;

  // Hopping: -t * sum_{<i,j>,sigma} (c^dag_i c_j + h.c.)
  for (auto [i, j] : bonds) {
    for (auto sigma : {Operator::Spin::Up, Operator::Spin::Down}) {
      H += hopping({-t, 0.0}, i, j, sigma);
    }
  }

  // On-site interaction: U * sum_i n_{i,up} n_{i,down}
  for (size_t i = 0; i < num_sites; ++i) {
    H += Expression(density_density(Operator::Spin::Up, i, Operator::Spin::Down, i)) *
         std::complex<double>(U, 0.0);
  }

  // Chemical potential: -mu * sum_i (n_{i,up} + n_{i,down})
  for (size_t i = 0; i < num_sites; ++i) {
    for (auto sigma : {Operator::Spin::Up, Operator::Spin::Down}) {
      H += Expression(density(sigma, i)) * std::complex<double>(-mu, 0.0);
    }
  }

  return H;
}

// ---------------------------------------------------------------------------
// Sector diagonalization
// ---------------------------------------------------------------------------

struct SectorResult {
  arma::vec eigenvalues;
  arma::cx_mat eigenvectors;
};

SectorResult diagonalize_sector(const Basis& basis, const Expression& H) {
  arma::cx_mat mat = compute_matrix_elements<arma::cx_mat>(basis, H);

  SectorResult result;
  if (!arma::eig_sym(result.eigenvalues, result.eigenvectors, mat)) {
    std::cerr << "Diagonalization failed.\n";
    std::exit(1);
  }
  return result;
}

// ---------------------------------------------------------------------------
// Thermal helpers
// ---------------------------------------------------------------------------

struct ThermalWeights {
  arma::vec weights;  // normalized exp(-beta E_n) / Z
  double logZ = 0.0;
  double free_energy = 0.0;
};

ThermalWeights compute_thermal_weights(const arma::vec& evals, double beta) {
  ThermalWeights w;
  const size_t n = evals.n_elem;
  w.weights.set_size(n);
  w.weights.zeros();

  if (n == 0) {
    w.logZ = -std::numeric_limits<double>::infinity();
    w.free_energy = 0.0;
    return w;
  }

  const double E0 = evals(0);
  double z_shift = 1.0;
  for (size_t k = 1; k < n; ++k) {
    z_shift += std::exp(-beta * (evals(k) - E0));
  }

  const double log_z_shift = std::log(z_shift);
  w.logZ = -beta * E0 + log_z_shift;
  w.free_energy = E0 - (1.0 / beta) * log_z_shift;

  for (size_t k = 0; k < n; ++k) {
    w.weights(k) = std::exp(-beta * (evals(k) - E0)) / z_shift;
  }

  return w;
}

// ---------------------------------------------------------------------------
// BRG step result
// ---------------------------------------------------------------------------

struct BrgStepResult {
  double t_prime;
  double U_prime;
  double mu_prime;
  double K_prime;

  // Diagnostics
  double E1;  // ground energy of (N=0, Sz=0)
  double E2;  // ground energy of (N=2, Sz=0)
  double E3;  // ground energy of (N=1, Sz=+1/2)
  double E4;  // ground energy of (N=1, Sz=-1/2)

  double F1;  // free energy of (N=0, Sz=0)
  double F2;  // free energy of (N=2, Sz=0)
  double F3;  // free energy of (N=1, Sz=+1/2)
  double F4;  // free energy of (N=1, Sz=-1/2)

  double lambda_avg;        // average lambda amplitude
  double lambda_sq_avg;     // average lambda^2 used for t'
  double lambda_spin_diff;  // |lambda_up - lambda_down| max
  double lambda_site_diff;  // |lambda_site1 - lambda_site3| max
  double closure_error;     // max closure check deviation
};

// ---------------------------------------------------------------------------
// Single BRG step (T=0)
// ---------------------------------------------------------------------------

BrgStepResult brg_step_zero_t(double t, double U, double mu) {
  constexpr size_t num_sites = 4;
  constexpr double nu = 2.0;  // number of inter-block couplings per direction

  const Expression H = build_block_hamiltonian(t, U, mu);

  Basis basis_N0 = Basis::with_fixed_particle_number_and_spin(num_sites, 0, 0);
  Basis basis_N1_up = Basis::with_fixed_particle_number_and_spin(num_sites, 1, 1);
  Basis basis_N1_down = Basis::with_fixed_particle_number_and_spin(num_sites, 1, -1);
  Basis basis_N2 = Basis::with_fixed_particle_number_and_spin(num_sites, 2, 0);

  SectorResult res_N0 = diagonalize_sector(basis_N0, H);
  SectorResult res_N1_up = diagonalize_sector(basis_N1_up, H);
  SectorResult res_N1_down = diagonalize_sector(basis_N1_down, H);
  SectorResult res_N2 = diagonalize_sector(basis_N2, H);

  const double E1 = res_N0.eigenvalues(0);
  const double E2 = res_N2.eigenvalues(0);
  const double E3 = res_N1_up.eigenvalues(0);
  const double E4 = res_N1_down.eigenvalues(0);

  const arma::cx_vec& psi_0 = res_N0.eigenvectors.col(0);
  const arma::cx_vec& psi_up = res_N1_up.eigenvectors.col(0);
  const arma::cx_vec& psi_down = res_N1_down.eigenvectors.col(0);
  const arma::cx_vec& psi_ud = res_N2.eigenvectors.col(0);

  const double U_prime = E1 + E2 - 2.0 * E3;
  const double mu_prime = E1 - E3;
  const double K_prime = E1;

  const size_t border_sites[] = {1, 3};

  double lambda_sum = 0.0;
  int lambda_count = 0;
  double max_spin_diff = 0.0;
  double max_site_diff = 0.0;
  double max_closure_error = 0.0;

  double lambdas[2][2] = {};   // [site_idx][spin_idx]
  double closures[2][2] = {};  // [site_idx][spin_idx]

  for (size_t si = 0; si < 2; ++si) {
    const size_t site = border_sites[si];

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

      const Basis& closure_col_basis = (spi == 0) ? basis_N1_down : basis_N1_up;
      const arma::cx_vec& psi_minus_sigma = (spi == 0) ? psi_down : psi_up;

      arma::cx_mat M_closure =
          compute_rectangular_matrix_elements<arma::cx_mat>(basis_N2, closure_col_basis, c_dag);
      std::complex<double> closure_val = arma::cdot(psi_ud, M_closure * psi_minus_sigma);

      closures[si][spi] = std::abs(closure_val);
    }
  }

  const double lambda_avg = lambda_sum / lambda_count;

  for (size_t si = 0; si < 2; ++si) {
    double diff = std::abs(lambdas[si][0] - lambdas[si][1]);
    max_spin_diff = std::max(max_spin_diff, diff);
  }

  for (size_t spi = 0; spi < 2; ++spi) {
    double diff = std::abs(lambdas[0][spi] - lambdas[1][spi]);
    max_site_diff = std::max(max_site_diff, diff);
  }

  for (size_t si = 0; si < 2; ++si) {
    for (size_t spi = 0; spi < 2; ++spi) {
      double err = std::abs(lambdas[si][spi] - closures[si][spi]);
      max_closure_error = std::max(max_closure_error, err);
    }
  }

  const double t_prime = nu * lambda_avg * lambda_avg * t;
  const double lambda_sq_avg = lambda_avg * lambda_avg;

  return BrgStepResult{t_prime,
                       U_prime,
                       mu_prime,
                       K_prime,
                       E1,
                       E2,
                       E3,
                       E4,
                       E1,
                       E2,
                       E3,
                       E4,
                       lambda_avg,
                       lambda_sq_avg,
                       max_spin_diff,
                       max_site_diff,
                       max_closure_error};
}

// ---------------------------------------------------------------------------
// Single BRG step (finite T)
// ---------------------------------------------------------------------------

BrgStepResult brg_step_finite_t(double t, double U, double mu, double T) {
  if (use_zero_temperature(T)) {
    return brg_step_zero_t(t, U, mu);
  }

  constexpr size_t num_sites = 4;
  constexpr double nu = 2.0;  // number of inter-block couplings per direction

  const Expression H = build_block_hamiltonian(t, U, mu);

  Basis basis_N0 = Basis::with_fixed_particle_number_and_spin(num_sites, 0, 0);
  Basis basis_N1_up = Basis::with_fixed_particle_number_and_spin(num_sites, 1, 1);
  Basis basis_N1_down = Basis::with_fixed_particle_number_and_spin(num_sites, 1, -1);
  Basis basis_N2 = Basis::with_fixed_particle_number_and_spin(num_sites, 2, 0);

  SectorResult res_N0 = diagonalize_sector(basis_N0, H);
  SectorResult res_N1_up = diagonalize_sector(basis_N1_up, H);
  SectorResult res_N1_down = diagonalize_sector(basis_N1_down, H);
  SectorResult res_N2 = diagonalize_sector(basis_N2, H);

  const double E1 = res_N0.eigenvalues(0);
  const double E2 = res_N2.eigenvalues(0);
  const double E3 = res_N1_up.eigenvalues(0);
  const double E4 = res_N1_down.eigenvalues(0);

  const double beta = 1.0 / T;
  const ThermalWeights w_N0 = compute_thermal_weights(res_N0.eigenvalues, beta);
  const ThermalWeights w_N1_up = compute_thermal_weights(res_N1_up.eigenvalues, beta);
  const ThermalWeights w_N1_down = compute_thermal_weights(res_N1_down.eigenvalues, beta);
  const ThermalWeights w_N2 = compute_thermal_weights(res_N2.eigenvalues, beta);

  const double F1 = w_N0.free_energy;
  const double F2 = w_N2.free_energy;
  const double F3 = w_N1_up.free_energy;
  const double F4 = w_N1_down.free_energy;

  const double U_prime = F1 + F2 - 2.0 * F3;
  const double mu_prime = F1 - F3;
  const double K_prime = F1;

  const arma::cx_vec& psi_0 = res_N0.eigenvectors.col(0);
  const arma::cx_mat& eigvecs_up = res_N1_up.eigenvectors;
  const arma::cx_mat& eigvecs_down = res_N1_down.eigenvectors;
  const arma::cx_mat& eigvecs_ud = res_N2.eigenvectors;

  const size_t border_sites[] = {1, 3};

  double lambda_sum = 0.0;
  double lambda_sq_sum = 0.0;
  int lambda_count = 0;
  double max_spin_diff = 0.0;
  double max_site_diff = 0.0;
  double max_closure_error = 0.0;

  double lambdas[2][2] = {};   // lambda_th per site/spin
  double closures[2][2] = {};  // closure_th per site/spin

  for (size_t si = 0; si < 2; ++si) {
    const size_t site = border_sites[si];

    for (size_t spi = 0; spi < 2; ++spi) {
      auto sigma = (spi == 0) ? Operator::Spin::Up : Operator::Spin::Down;

      Expression c_dag(creation(sigma, site));

      const Basis& row_basis = (spi == 0) ? basis_N1_up : basis_N1_down;
      const arma::cx_mat& eigvecs_sigma = (spi == 0) ? eigvecs_up : eigvecs_down;
      const arma::vec& weights_sigma = (spi == 0) ? w_N1_up.weights : w_N1_down.weights;

      arma::cx_mat M =
          compute_rectangular_matrix_elements<arma::cx_mat>(row_basis, basis_N0, c_dag);
      arma::cx_vec v = M * psi_0;
      arma::cx_vec overlaps = eigvecs_sigma.st() * v;

      arma::vec overlap_sq = arma::square(arma::abs(overlaps));
      double lambda_sq = arma::dot(weights_sigma, overlap_sq);
      double lambda_th = std::sqrt(lambda_sq);

      lambdas[si][spi] = lambda_th;
      lambda_sum += lambda_th;
      lambda_sq_sum += lambda_sq;
      ++lambda_count;

      const Basis& closure_col_basis = (spi == 0) ? basis_N1_down : basis_N1_up;
      const arma::cx_mat& eigvecs_minus_sigma = (spi == 0) ? eigvecs_down : eigvecs_up;
      const arma::vec& weights_minus_sigma = (spi == 0) ? w_N1_down.weights : w_N1_up.weights;

      arma::cx_mat M_closure =
          compute_rectangular_matrix_elements<arma::cx_mat>(basis_N2, closure_col_basis, c_dag);
      arma::cx_mat A = eigvecs_ud.st() * M_closure * eigvecs_minus_sigma;
      arma::mat A_sq = arma::square(arma::abs(A));
      double closure_sq = arma::dot(w_N2.weights, A_sq * weights_minus_sigma);
      double closure_th = std::sqrt(closure_sq);

      closures[si][spi] = closure_th;
    }
  }

  const double lambda_avg = lambda_sum / lambda_count;
  const double lambda_sq_avg = lambda_sq_sum / lambda_count;

  for (size_t si = 0; si < 2; ++si) {
    double diff = std::abs(lambdas[si][0] - lambdas[si][1]);
    max_spin_diff = std::max(max_spin_diff, diff);
  }

  for (size_t spi = 0; spi < 2; ++spi) {
    double diff = std::abs(lambdas[0][spi] - lambdas[1][spi]);
    max_site_diff = std::max(max_site_diff, diff);
  }

  for (size_t si = 0; si < 2; ++si) {
    for (size_t spi = 0; spi < 2; ++spi) {
      double err = std::abs(lambdas[si][spi] - closures[si][spi]);
      max_closure_error = std::max(max_closure_error, err);
    }
  }

  const double t_prime = nu * lambda_sq_avg * t;

  return BrgStepResult{t_prime,
                       U_prime,
                       mu_prime,
                       K_prime,
                       E1,
                       E2,
                       E3,
                       E4,
                       F1,
                       F2,
                       F3,
                       F4,
                       lambda_avg,
                       lambda_sq_avg,
                       max_spin_diff,
                       max_site_diff,
                       max_closure_error};
}

// ---------------------------------------------------------------------------
// Mode B: tune mu for quarter filling (N=1 per block)
// ---------------------------------------------------------------------------

// Returns the tuned mu, or the fallback mu if the N=1 window has collapsed.
// Sets window_exists to indicate whether a valid window was found.
double tune_mu_for_quarter_filling_zero_t(double t, double U, bool& window_exists) {
  constexpr size_t num_sites = 4;

  const Expression H0 = build_block_hamiltonian(t, U, 0.0);

  Basis basis_N0 = Basis::with_fixed_particle_number_and_spin(num_sites, 0, 0);
  Basis basis_N1_up = Basis::with_fixed_particle_number_and_spin(num_sites, 1, 1);
  Basis basis_N1_down = Basis::with_fixed_particle_number_and_spin(num_sites, 1, -1);
  Basis basis_N2 = Basis::with_fixed_particle_number_and_spin(num_sites, 2, 0);

  SectorResult res_N0 = diagonalize_sector(basis_N0, H0);
  SectorResult res_N1_up = diagonalize_sector(basis_N1_up, H0);
  SectorResult res_N1_down = diagonalize_sector(basis_N1_down, H0);
  SectorResult res_N2 = diagonalize_sector(basis_N2, H0);

  const double e0 = res_N0.eigenvalues(0);
  const double e1 = std::min(res_N1_up.eigenvalues(0), res_N1_down.eigenvalues(0));
  const double e2 = res_N2.eigenvalues(0);

  const double mu_low = e1 - e0;
  const double mu_high = e2 - e1;

  if (mu_low < mu_high) {
    window_exists = true;
    return (mu_low + mu_high) / 2.0;
  }

  window_exists = false;
  return (mu_low + mu_high) / 2.0;
}

double tune_mu_for_quarter_filling_finite_t(double t, double U, double T, bool& window_exists) {
  if (use_zero_temperature(T)) {
    return tune_mu_for_quarter_filling_zero_t(t, U, window_exists);
  }

  constexpr size_t num_sites = 4;

  const Expression H0 = build_block_hamiltonian(t, U, 0.0);

  Basis basis_N0 = Basis::with_fixed_particle_number_and_spin(num_sites, 0, 0);
  Basis basis_N1_up = Basis::with_fixed_particle_number_and_spin(num_sites, 1, 1);
  Basis basis_N1_down = Basis::with_fixed_particle_number_and_spin(num_sites, 1, -1);
  Basis basis_N2 = Basis::with_fixed_particle_number_and_spin(num_sites, 2, 0);

  SectorResult res_N0 = diagonalize_sector(basis_N0, H0);
  SectorResult res_N1_up = diagonalize_sector(basis_N1_up, H0);
  SectorResult res_N1_down = diagonalize_sector(basis_N1_down, H0);
  SectorResult res_N2 = diagonalize_sector(basis_N2, H0);

  const double beta = 1.0 / T;
  const ThermalWeights w_N0 = compute_thermal_weights(res_N0.eigenvalues, beta);
  const ThermalWeights w_N1_up = compute_thermal_weights(res_N1_up.eigenvalues, beta);
  const ThermalWeights w_N1_down = compute_thermal_weights(res_N1_down.eigenvalues, beta);
  const ThermalWeights w_N2 = compute_thermal_weights(res_N2.eigenvalues, beta);

  const double F0 = w_N0.free_energy;
  const double F1 = 0.5 * (w_N1_up.free_energy + w_N1_down.free_energy);
  const double F2 = w_N2.free_energy;

  const double mu_low = F1 - F0;
  const double mu_high = F2 - F1;

  if (mu_low < mu_high) {
    window_exists = true;
    return (mu_low + mu_high) / 2.0;
  }

  window_exists = false;
  return (mu_low + mu_high) / 2.0;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main(int argc, char** argv) {
  CliOptions opts = parse_cli_options(argc, argv);

  const bool mode_b = (opts.mode == "B" || opts.mode == "b");
  const bool temp_rescaled = (opts.temp_mode == "rescaled");

  const double initial_t_over_T = (std::abs(opts.t) > 0.0) ? (opts.temperature / opts.t) : 0.0;

  std::cerr << "=== Block RG for 2D Hubbard (2x2 blocks, finite T) ===" << std::endl;
  std::cerr << "Mode: " << (mode_b ? "B (fixed quarter filling)" : "A (density flows)")
            << std::endl;
  std::cerr << "Temp mode: " << (temp_rescaled ? "rescaled" : "fixed") << " (T=" << opts.temperature
            << ")" << std::endl;
  std::cerr << "Initial: t=" << opts.t << ", U=" << opts.U << ", mu=" << opts.mu << std::endl;
  std::cerr << std::endl;

  std::cout << std::setprecision(8) << std::fixed;
  std::cout << "# iter      t             U             mu            U/t           mu/t"
            << "          lambda        spin_diff     site_diff     closure_err"
            << "       T             T/t           F1            F2            F3            F4";
  if (mode_b) {
    std::cout << "   window";
  }
  std::cout << "\n";

  double t = opts.t;
  double U = opts.U;
  double mu = opts.mu;

  for (int n = 0; n < opts.iterations; ++n) {
    bool window_exists = true;

    const double T_iter = temp_rescaled ? (initial_t_over_T * t) : opts.temperature;

    if (mode_b) {
      mu = tune_mu_for_quarter_filling_finite_t(t, U, T_iter, window_exists);
    }

    BrgStepResult result = brg_step_finite_t(t, U, mu, T_iter);

    const double T_over_t = (std::abs(t) > 0.0) ? (T_iter / t) : 0.0;

    std::cout << std::setw(4) << n << "  " << std::setw(13) << t << " " << std::setw(13) << U << " "
              << std::setw(13) << mu << " " << std::setw(13) << (U / t) << " " << std::setw(13)
              << (mu / t) << " " << std::setw(13) << result.lambda_avg << " " << std::setw(13)
              << result.lambda_spin_diff << " " << std::setw(13) << result.lambda_site_diff << " "
              << std::setw(13) << result.closure_error << " " << std::setw(13) << T_iter << " "
              << std::setw(13) << T_over_t << " " << std::setw(13) << result.F1 << " "
              << std::setw(13) << result.F2 << " " << std::setw(13) << result.F3 << " "
              << std::setw(13) << result.F4;
    if (mode_b) {
      std::cout << "   " << (window_exists ? "yes" : "NO");
    }
    std::cout << "\n";

    std::cerr << "Iteration " << n << ": T=" << T_iter << " E1=" << result.E1 << " E2=" << result.E2
              << " E3=" << result.E3 << " E4=" << result.E4 << std::endl;
    std::cerr << "  F1=" << result.F1 << " F2=" << result.F2 << " F3=" << result.F3
              << " F4=" << result.F4 << std::endl;
    std::cerr << "  |E3-E4| = " << std::abs(result.E3 - result.E4) << " (spin degeneracy check)"
              << std::endl;
    std::cerr << "  U'=" << result.U_prime << " mu'=" << result.mu_prime << " K'=" << result.K_prime
              << " t'=" << result.t_prime << std::endl;
    std::cerr << "  lambda=" << result.lambda_avg << " closure_err=" << result.closure_error
              << std::endl;
    std::cerr << std::endl;

    const double Ut_ratio = std::abs(U / t);
    if (Ut_ratio > 1e6) {
      std::cerr << "Divergence detected: |U/t| > 1e6. Stopping.\n";
      break;
    }
    if (std::abs(result.t_prime) < 1e-15) {
      std::cerr << "t' ~ 0. Stopping.\n";
      break;
    }

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

    t = result.t_prime;
    U = result.U_prime;
    mu = mode_b ? mu : result.mu_prime;
  }

  std::cerr << "Final parameters: t=" << t << ", U=" << U << ", mu=" << mu << ", U/t=" << (U / t)
            << std::endl;

  return 0;
}
