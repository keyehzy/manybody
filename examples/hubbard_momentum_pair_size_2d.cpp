#include <omp.h>

#include <armadillo>
#include <cereal/archives/binary.hpp>
#include <cmath>
#include <cstddef>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>

#include "algebra/fermion/basis.h"
#include "algebra/fermion/matrix_elements.h"
#include "algebra/model/hubbard_model_momentum.h"
#include "cxxopts.hpp"
#include "numerics/lanczos.h"
#include "numerics/linear_operator.h"
#include "utils/arma_cereal.h"

struct CliOptions {
  size_t size_x = 2;
  size_t size_y = 2;
  size_t particles = 4;
  int spin_projection = 0;
  size_t kx = 0;
  size_t ky = 0;
  double t = 1.0;
  double U = 4.0;
  size_t lanczos_steps = 100;
  std::string cache_dir = "cache";
};

CliOptions parse_cli_options(int argc, char** argv) {
  CliOptions o;

  cxxopts::Options cli(
      "hubbard_momentum_pair_size_2d",
      "Compute RMS pair separation using opposite-spin density-density correlator (2D)");
  // clang-format off
  cli.add_options()
      ("x,size-x", "Lattice size in x dimension",       cxxopts::value(o.size_x)->default_value("2"))
      ("y,size-y", "Lattice size in y dimension",       cxxopts::value(o.size_y)->default_value("2"))
      ("N,particles", "Number of particles",            cxxopts::value(o.particles)->default_value("4"))
      ("S,spin", "Spin projection (n_up - n_down)",     cxxopts::value(o.spin_projection)->default_value("0"))
      ("kx", "Total momentum Kx component",             cxxopts::value(o.kx)->default_value("0"))
      ("ky", "Total momentum Ky component",             cxxopts::value(o.ky)->default_value("0"))
      ("t,hopping", "Hopping amplitude",                cxxopts::value(o.t)->default_value("1.0"))
      ("U,interaction", "On-site interaction strength", cxxopts::value(o.U)->default_value("4.0"))
      ("k,lanczos-steps", "Lanczos iteration steps",    cxxopts::value(o.lanczos_steps)->default_value("100"))
      ("c,cache-dir", "Directory for caching matrices", cxxopts::value(o.cache_dir)->default_value("cache"))
      ("h,help", "Print usage");
  // clang-format on

  try {
    auto result = cli.parse(argc, argv);
    if (result.count("help")) {
      std::cout << cli.help() << "\n";
      std::exit(0);
    }
    std::cerr << "--- Parsed arguments ---" << std::endl;
    for (const auto& kv : result.arguments()) {
      std::cerr << "  " << kv.key() << ": " << kv.value() << std::endl;
    }
  } catch (const std::exception& e) {
    std::cerr << "Argument error: " << e.what() << "\n" << cli.help() << "\n";
    std::exit(1);
  }

  return o;
}

void validate_options(const CliOptions& opts) {
  if (opts.size_x == 0 || opts.size_y == 0) {
    std::cerr << "Lattice dimensions must be positive.\n";
    std::exit(1);
  }
  if (opts.particles == 0) {
    std::cerr << "Number of particles must be positive.\n";
    std::exit(1);
  }
  if (opts.lanczos_steps == 0) {
    std::cerr << "Lanczos steps must be positive.\n";
    std::exit(1);
  }
  const size_t sites = opts.size_x * opts.size_y;
  if (opts.particles > 2 * sites) {
    std::cerr << "Too many particles for the lattice size.\n";
    std::exit(1);
  }
  const int64_t particles_signed = static_cast<int64_t>(opts.particles);
  const int64_t spin = static_cast<int64_t>(opts.spin_projection);
  if (std::abs(spin) > particles_signed) {
    std::cerr << "Spin projection magnitude cannot exceed particle count.\n";
    std::exit(1);
  }
  if ((particles_signed + spin) % 2 != 0) {
    std::cerr << "Invalid spin projection: (particles + spin) must be even.\n";
    std::exit(1);
  }
  if (opts.kx >= opts.size_x || opts.ky >= opts.size_y) {
    std::cerr << "Momentum components must be less than corresponding lattice dimensions.\n";
    std::exit(1);
  }
}

namespace {

// Generate cache file path for Hamiltonian components (kinetic or interaction)
std::string hamiltonian_cache_path(const CliOptions& opts, const std::string& component,
                                   const std::vector<size_t>& K) {
  std::ostringstream ss;
  ss << opts.cache_dir << "/hubbard_" << component << "_L=" << opts.size_x << "x" << opts.size_y
     << "_N=" << opts.particles << "_S=" << opts.spin_projection << "_K=" << K[0] << "," << K[1]
     << ".bin";
  return ss.str();
}

// Save a sparse matrix to file
void save_sparse_matrix(const std::string& path, const arma::sp_cx_mat& mat) {
  std::ofstream ofs(path, std::ios::binary);
  if (!ofs) {
    throw std::runtime_error("Failed to open file for writing: " + path);
  }
  cereal::BinaryOutputArchive archive(ofs);
  archive(mat);
}

// Load a sparse matrix from file
arma::sp_cx_mat load_sparse_matrix(const std::string& path) {
  std::ifstream ifs(path, std::ios::binary);
  if (!ifs) {
    throw std::runtime_error("Failed to open file for reading: " + path);
  }
  arma::sp_cx_mat mat;
  cereal::BinaryInputArchive archive(ifs);
  archive(mat);
  return mat;
}

// Load or compute a Hamiltonian component matrix
arma::sp_cx_mat load_or_compute_hamiltonian_component(const CliOptions& opts,
                                                      const std::string& component,
                                                      const std::vector<size_t>& K,
                                                      const Basis& basis, const Expression& expr) {
  const std::string path = hamiltonian_cache_path(opts, component, K);

  if (std::filesystem::exists(path)) {
    std::cerr << "Loading cached " << component << " matrix from " << path << "..." << std::endl;
    return load_sparse_matrix(path);
  }

  std::cerr << "Computing " << component << " matrix..." << std::endl;
  arma::sp_cx_mat mat = compute_matrix_elements<arma::sp_cx_mat>(basis, expr);

  std::filesystem::create_directories(opts.cache_dir);
  std::cerr << "Saving " << component << " matrix to " << path << "..." << std::endl;
  save_sparse_matrix(path, mat);

  return mat;
}

// Compute minimum image distance squared for periodic boundary conditions (2D)
double min_image_distance_squared(const std::vector<size_t>& r, const std::vector<size_t>& L) {
  double r_squared = 0.0;
  for (size_t d = 0; d < r.size(); ++d) {
    // Minimum image convention: wrapped distance in [-L/2, L/2]
    const double r_d = static_cast<double>(r[d]);
    const double L_d = static_cast<double>(L[d]);
    const double wrapped = std::min(r_d, L_d - r_d);
    r_squared += wrapped * wrapped;
  }
  return r_squared;
}

}  // namespace

int main(int argc, char** argv) {
  const CliOptions opts = parse_cli_options(argc, argv);
  validate_options(opts);

  const std::vector<size_t> size = {opts.size_x, opts.size_y};
  const size_t sites = opts.size_x * opts.size_y;
  const std::vector<size_t> K = {opts.kx, opts.ky};

  std::cerr << "Setting up 2D Hubbard model..." << std::endl;
  // Create a "base" model with t=1, u=1 for computing cacheable matrices
  HubbardModelMomentum hubbard_base(1.0, 1.0, size);
  Index index(size);

  // Build basis
  std::cerr << "Building basis for sector K=(" << K[0] << "," << K[1] << ")..." << std::endl;
  Basis basis = Basis::with_fixed_particle_number_spin_momentum(sites, opts.particles,
                                                                opts.spin_projection, index, K);

  if (basis.set.empty()) {
    std::cerr << "No basis states for the chosen particle number, spin, and momentum.\n";
    return 1;
  }

  std::cerr << "Basis size: " << basis.set.size() << std::endl;

  // Load or compute kinetic and interaction matrices
  std::cerr << "Building Hamiltonian matrices..." << std::endl;
  const Expression kinetic_expr = hubbard_base.kinetic();
  const Expression interaction_expr = hubbard_base.interaction();

  arma::sp_cx_mat kinetic =
      load_or_compute_hamiltonian_component(opts, "kinetic", K, basis, kinetic_expr);
  arma::sp_cx_mat interaction =
      load_or_compute_hamiltonian_component(opts, "interaction", K, basis, interaction_expr);

  // Assemble full Hamiltonian: H = t*kinetic + U*interaction
  std::cerr << "Assembling Hamiltonian with t=" << opts.t << ", U=" << opts.U << "..." << std::endl;
  arma::sp_cx_mat H = opts.t * kinetic + opts.U * interaction;

  SparseComplexMatrixOperator op(std::move(H));
  const size_t dimension = op.dimension();
  const size_t lanczos_steps = std::min(opts.lanczos_steps, dimension);

  // Find ground state
  std::cerr << "Finding ground state using Lanczos (" << lanczos_steps << " steps)..." << std::endl;
  const auto eigenpair = find_min_eigenpair(op, lanczos_steps);
  const arma::cx_vec& psi = eigenpair.vector;

  const arma::cx_vec residual = op.apply(psi) - eigenpair.value * psi;
  const double residual_norm = arma::norm(residual);

  std::cerr << "Ground state energy: " << eigenpair.value << std::endl;
  std::cerr << "Residual norm: " << residual_norm << std::endl;

  // Compute particle numbers for connected correlator
  // N_up = (N + Sz) / 2, N_down = (N - Sz) / 2
  const int64_t particles_signed = static_cast<int64_t>(opts.particles);
  const int64_t spin = static_cast<int64_t>(opts.spin_projection);
  const double n_up = static_cast<double>((particles_signed + spin) / 2);
  const double n_down = static_cast<double>((particles_signed - spin) / 2);
  const double N = static_cast<double>(sites);

  // Background term for connected correlator: <n_up><n_down> = (N_up/N)(N_down/N)
  const double background = (n_up / N) * (n_down / N);

  std::cerr << "N_up = " << n_up << ", N_down = " << n_down << ", N = " << N << std::endl;
  std::cerr << "Background <n_up><n_down> = " << background << std::endl;

  // Compute G_{up,down}(r) for all separations r
  std::cerr << "\nComputing opposite-spin correlator for all separations on "
            << omp_get_max_threads() << " threads..." << std::endl;

  // Structure to hold results for each separation
  struct CorrelationResult {
    size_t rx, ry;
    double r_squared;
    double G_r;
    double G_conn_r;
  };

  const size_t total_sites = opts.size_x * opts.size_y;
  std::vector<CorrelationResult> results(total_sites);

  double sum_G = 0.0;          // Sigma_r G(r)
  double sum_r2_G = 0.0;       // Sigma_r r^2 G(r)
  double sum_G_conn = 0.0;     // Sigma_r G_conn(r)
  double sum_r2_G_conn = 0.0;  // Sigma_r r^2 G_conn(r)

#pragma omp parallel for schedule(dynamic) reduction(+ : sum_G, sum_r2_G, sum_G_conn, sum_r2_G_conn)
  for (size_t idx = 0; idx < total_sites; ++idx) {
    // Convert flat index to (rx, ry)
    const size_t rx = idx / opts.size_y;
    const size_t ry = idx % opts.size_y;
    const std::vector<size_t> r = {rx, ry};

    // Compute correlation operator (sparse, no caching needed)
    const Expression corr_expr = hubbard_base.opposite_spin_correlation(r);
    arma::sp_cx_mat G_op = compute_matrix_elements<arma::sp_cx_mat>(basis, corr_expr);

    // Compute expectation value <psi|G(r)|psi>
    const arma::cx_vec G_psi = G_op * psi;
    const std::complex<double> G_r_complex = arma::cdot(psi, G_psi);
    const double G_r = G_r_complex.real();

    // Connected correlator
    const double G_conn_r = G_r - background;

    // Minimum image distance squared
    const double r_squared = min_image_distance_squared(r, size);

    // Accumulate sums (handled by OpenMP reduction)
    sum_G += G_r;
    sum_r2_G += r_squared * G_r;
    sum_G_conn += G_conn_r;
    sum_r2_G_conn += r_squared * G_conn_r;

    // Store result for ordered output
    results[idx] = {rx, ry, r_squared, G_r, G_conn_r};
  }

  // Output results in order
  std::cout << std::setprecision(10);
  std::cout << "# rx ry r^2 G(r) G_conn(r)\n";
  for (const auto& res : results) {
    std::cout << res.rx << " " << res.ry << " " << res.r_squared << " " << res.G_r << " "
              << res.G_conn_r << "\n";
  }

  // Compute RMS separations
  // xi^2 = Sigma_r r^2 G(r) / Sigma_r G(r)
  const double xi_squared = (sum_G > 0.0) ? (sum_r2_G / sum_G) : 0.0;
  const double xi = std::sqrt(xi_squared);

  // xi^2_conn = Sigma_r r^2 G_conn(r) / Sigma_r G_conn(r)
  const double xi_conn_squared = (sum_G_conn > 0.0) ? (sum_r2_G_conn / sum_G_conn) : 0.0;
  const double xi_conn = std::sqrt(std::abs(xi_conn_squared));

  std::cerr << "\n=== Results ===" << std::endl;
  std::cerr << "Lattice: " << opts.size_x << "x" << opts.size_y << std::endl;
  std::cerr << "Particles: " << opts.particles << ", Sz: " << opts.spin_projection << std::endl;
  std::cerr << "t=" << opts.t << ", U=" << opts.U << ", U/t=" << (opts.U / opts.t) << std::endl;
  std::cerr << "Ground state energy: " << eigenpair.value << std::endl;
  std::cerr << std::endl;

  std::cerr << "Full correlator:" << std::endl;
  std::cerr << "  Sigma_r G(r) = " << sum_G << std::endl;
  std::cerr << "  Sigma_r r^2 G(r) = " << sum_r2_G << std::endl;
  std::cerr << "  xi^2 = " << xi_squared << std::endl;
  std::cerr << "  xi = " << xi << " (lattice spacings)" << std::endl;
  std::cerr << std::endl;

  std::cerr << "Connected correlator (G - <n_up><n_down>):" << std::endl;
  std::cerr << "  Background = " << background << std::endl;
  std::cerr << "  Sigma_r G_conn(r) = " << sum_G_conn << std::endl;
  std::cerr << "  Sigma_r r^2 G_conn(r) = " << sum_r2_G_conn << std::endl;
  std::cerr << "  xi^2_conn = " << xi_conn_squared << std::endl;
  std::cerr << "  xi_conn = " << xi_conn << " (lattice spacings)" << std::endl;

  return 0;
}
