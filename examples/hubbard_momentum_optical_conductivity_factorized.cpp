#include <omp.h>

#include <armadillo>
#include <complex>
#include <cstddef>
#include <cstdlib>
#include <ctime>
#include <exception>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#include "algebra/basis.h"
#include "algebra/matrix_elements.h"
#include "algebra/model/hubbard_model_momentum.h"
#include "algorithms/optical_conductivity.h"
#include "cxxopts.hpp"
#include "numerics/evolve_state.h"
#include "numerics/hubbard_momentum_factorized.h"

struct CliOptions {
  size_t size_x = 2;
  size_t size_y = 2;
  size_t size_z = 2;
  size_t particles = 4;
  int spin_projection = 0;
  size_t qx = 1;
  size_t qy = 0;
  size_t qz = 0;
  size_t direction = 0;
  double t = 1.0;
  double U = 4.0;
  double beta = 10.0;
  double dt = 0.01;
  size_t krylov_steps = 20;
  size_t steps = 1000;
  size_t num_samples = 5;
};

CliOptions parse_cli_options(int argc, char** argv) {
  CliOptions o;

  cxxopts::Options cli(
      "hubbard_momentum_optical_conductivity",
      "Compute optical conductivity using the factorized momentum-space Hubbard model");
  // clang-format off
  cli.add_options()
      ("x,size-x", "Lattice size in x dimension",       cxxopts::value(o.size_x)->default_value("2"))
      ("y,size-y", "Lattice size in y dimension",       cxxopts::value(o.size_y)->default_value("2"))
      ("z,size-z", "Lattice size in z dimension",       cxxopts::value(o.size_z)->default_value("2"))
      ("N,particles", "Number of particles",            cxxopts::value(o.particles)->default_value("4"))
      ("S,spin", "Spin projection (n_up - n_down)",     cxxopts::value(o.spin_projection)->default_value("0"))
      ("qx", "Transfer momentum qx component",          cxxopts::value(o.qx)->default_value("1"))
      ("qy", "Transfer momentum qy component",          cxxopts::value(o.qy)->default_value("0"))
      ("qz", "Transfer momentum qz component",          cxxopts::value(o.qz)->default_value("0"))
      ("D,direction", "Current operator direction",     cxxopts::value(o.direction)->default_value("0"))
      ("t,hopping", "Hopping amplitude",                cxxopts::value(o.t)->default_value("1.0"))
      ("U,interaction", "On-site interaction strength", cxxopts::value(o.U)->default_value("4.0"))
      ("b,beta", "Inverse temperature",                 cxxopts::value(o.beta)->default_value("10.0"))
      ("d,dt", "Real-time step size",                   cxxopts::value(o.dt)->default_value("0.01"))
      ("k,krylov-steps", "Krylov subspace dimension",   cxxopts::value(o.krylov_steps)->default_value("20"))
      ("s,steps", "Real-time evolution steps",          cxxopts::value(o.steps)->default_value("1000"))
      ("n,num-samples", "Number of stochastic samples", cxxopts::value(o.num_samples)->default_value("5"))
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
  if (opts.size_x == 0 || opts.size_y == 0 || opts.size_z == 0) {
    std::cerr << "Lattice dimensions must be positive.\n";
    std::exit(1);
  }
  if (opts.particles == 0) {
    std::cerr << "Number of particles must be positive.\n";
    std::exit(1);
  }
  const size_t sites = opts.size_x * opts.size_y * opts.size_z;
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
  if (opts.qx >= opts.size_x || opts.qy >= opts.size_y || opts.qz >= opts.size_z) {
    std::cerr << "Transfer momentum components must be less than lattice dimensions.\n";
    std::exit(1);
  }
  if (opts.direction >= 3) {
    std::cerr << "Direction must be 0, 1, or 2.\n";
    std::exit(1);
  }
  if (opts.beta <= 0.0) {
    std::cerr << "Inverse temperature must be positive.\n";
    std::exit(1);
  }
  if (opts.dt <= 0.0) {
    std::cerr << "Time step must be positive.\n";
    std::exit(1);
  }
  if (opts.krylov_steps == 0) {
    std::cerr << "Krylov steps must be positive.\n";
    std::exit(1);
  }
  if (opts.steps == 0) {
    std::cerr << "Steps must be positive.\n";
    std::exit(1);
  }
  if (opts.num_samples == 0) {
    std::cerr << "Number of samples must be positive.\n";
    std::exit(1);
  }
}

namespace {
arma::cx_vec random_complex_vector(size_t dimension, std::mt19937& rng) {
  std::normal_distribution<double> dist(0.0, 1.0);
  arma::cx_vec v(dimension);
  for (size_t i = 0; i < dimension; ++i) {
    v(i) = std::complex<double>(dist(rng), dist(rng));
  }
  return v;
}
}  // namespace

int main(int argc, char** argv) {
  const CliOptions opts = parse_cli_options(argc, argv);
  validate_options(opts);

  const std::vector<size_t> size = {opts.size_x, opts.size_y, opts.size_z};
  const size_t sites = opts.size_x * opts.size_y * opts.size_z;

  // Momentum transfer Q
  const std::vector<size_t> Q = {opts.qx, opts.qy, opts.qz};

  std::cerr << "Setting up Hubbard model..." << std::endl;
  // Factorized Hamiltonian uses the full momentum basis without fixed total K.
  Basis basis =
      Basis::with_fixed_particle_number_and_spin(sites, opts.particles, opts.spin_projection);
  if (basis.set.empty()) {
    std::cerr << "No basis states for the chosen particle number and spin.\n";
    return 1;
  }

  std::cerr << "Basis size: " << basis.set.size() << std::endl;

  HubbardMomentumFactorized hubbard(basis, size, opts.t, opts.U);
  HubbardModelMomentum hubbard_model(opts.t, opts.U, size);

  std::cerr << "Building current operator J(Q)..." << std::endl;
  const Expression current_expr = hubbard_model.current(Q, opts.direction);
  arma::sp_cx_mat J_Q = compute_matrix_elements<arma::sp_cx_mat>(basis, current_expr);
  arma::sp_cx_mat J_Q_adj = J_Q.t();

  std::cerr << "Current operator J(Q): " << J_Q.n_rows << "x" << J_Q.n_cols << " with "
            << J_Q.n_nonzero << " non-zeros" << std::endl;

  // Compute current-current correlator using stochastic trace estimation
  std::vector<std::complex<double>> correlator(opts.steps, std::complex<double>(0.0, 0.0));

  std::cerr << "Starting " << opts.num_samples << " stochastic samples on " << omp_get_max_threads()
            << " threads..." << std::endl;

#pragma omp parallel
  {
    const unsigned int seed = static_cast<unsigned int>(std::time(nullptr)) ^
                              (static_cast<unsigned int>(omp_get_thread_num()) << 16);
    std::mt19937 rng(seed);

    std::vector<std::complex<double>> thread_sum(opts.steps, std::complex<double>(0.0, 0.0));

    EvolutionOptions evolve_options;
    evolve_options.krylov_steps = opts.krylov_steps;

#pragma omp for schedule(static)
    for (int s = 0; s < static_cast<int>(opts.num_samples); ++s) {
      // Random initial vector in the full momentum basis
      arma::cx_vec v = random_complex_vector(basis.set.size(), rng);
      const double v_norm = arma::norm(v);
      if (v_norm > 0.0) {
        v /= v_norm;
      }

      // Apply imaginary-time evolution to get thermalized state: exp(-βH/2)|v>
      arma::cx_vec v_beta =
          imaginary_time_evolve_state(hubbard, v, 0.5 * opts.beta, evolve_options);

      // Partition function normalization
      const double Z = std::real(arma::cdot(v_beta, v_beta));

      // |φ₁> = |v_β>
      arma::cx_vec phi_1 = v_beta;

      // |φ₂> = J(Q)|v_β>
      arma::cx_vec phi_2 = J_Q * v_beta;

      // Time evolution and correlation function computation
      // C(t) = <φ₁(t)|J†(Q)|φ₂(t)>/Z = <v_β|e^{iHt} J†(Q) e^{-iHt} J(Q)|v_β>/Z
      for (size_t i = 0; i < opts.steps; ++i) {
        // Compute <φ₁|J†(Q)|φ₂>
        arma::cx_vec J_adj_phi_2 = J_Q_adj * phi_2;
        thread_sum[i] += arma::cdot(phi_1, J_adj_phi_2) / Z;

        // Evolve both states forward in time
        phi_1 = time_evolve_state(hubbard, phi_1, opts.dt, evolve_options);
        phi_2 = time_evolve_state(hubbard, phi_2, opts.dt, evolve_options);
      }

#pragma omp critical
      {
        std::cerr << "Sample " << (s + 1) << "/" << opts.num_samples << " completed by thread "
                  << omp_get_thread_num() << std::endl;
      }
    }

#pragma omp critical
    {
      for (size_t i = 0; i < opts.steps; ++i) {
        correlator[i] += thread_sum[i];
      }
    }
  }

  // Average over samples
  for (auto& c : correlator) {
    c /= static_cast<double>(opts.num_samples);
  }

  // Compute optical conductivity from correlation function
  const double volume = static_cast<double>(sites);
  const OpticalConductivityResult result =
      compute_optical_conductivity(correlator, opts.dt, opts.beta, volume);

  // Output results
  std::cout << std::setprecision(10);
  for (size_t i = 0; i < result.frequencies.size(); ++i) {
    std::cout << result.frequencies[i] << " " << result.sigma[i].real() << " "
              << result.sigma[i].imag() << "\n";
  }

  std::cerr << "\nOptical Sum Rule Check\n";
  std::cerr << "Integral Re[sigma(w)] dw: " << result.sum_rule << "\n";

  return 0;
}
