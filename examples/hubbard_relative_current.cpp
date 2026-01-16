#include <omp.h>

#include <armadillo>
#include <complex>
#include <cstddef>
#include <ctime>
#include <exception>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#include "cxxopts.hpp"
#include "hubbard_relative_operators.h"
#include "numerics/evolve_state.h"
#include "numerics/linear_operator.h"

struct CliOptions {
  size_t lattice_size = 8;
  size_t kx = 1;
  size_t ky = 0;
  size_t kz = 0;
  double t = 1.0;
  double U = -10.0;
  double beta = 10.0;
  double dt = 0.01;
  size_t krylov_steps = 20;
  size_t steps = 1000;
  size_t num_samples = 5;
};

namespace {
arma::cx_vec random_complex_vector(size_t dimension, std::mt19937& rng) {
  std::normal_distribution<double> dist(0.0, 1.0);
  arma::cx_vec v(dimension);
  for (size_t i = 0; i < dimension; ++i) {
    v(i) = std::complex<double>(dist(rng), dist(rng));
  }
  return v;
}

void parse_cli_options(int argc, char** argv, CliOptions* options_out) {
  cxxopts::Options options("hubbard_relative_current",
                           "Compute current-current correlator for the 3D relative Hubbard model");
  // clang-format off
  options.add_options()
      ("L,lattice-size", "Lattice size per dimension",     cxxopts::value<size_t>()->default_value("8"))
      ("x,kx", "Total momentum Kx component",              cxxopts::value<size_t>()->default_value("1"))
      ("y,ky", "Total momentum Ky component",              cxxopts::value<size_t>()->default_value("0"))
      ("z,kz", "Total momentum Kz component",              cxxopts::value<size_t>()->default_value("0"))
      ("t,hopping", "Hopping amplitude",                   cxxopts::value<double>()->default_value("1.0"))
      ("U,interaction", "On-site interaction strength",    cxxopts::value<double>()->default_value("-10.0"))
      ("b,beta", "Inverse temperature",                    cxxopts::value<double>()->default_value("10.0"))
      ("d,dt", "Real-time step size",                      cxxopts::value<double>()->default_value("0.01"))
      ("k,krylov-steps", "Krylov subspace dimension",      cxxopts::value<size_t>()->default_value("20"))
      ("s,steps", "Real-time evolution steps",             cxxopts::value<size_t>()->default_value("1000"))
      ("n,num-samples", "Number of stochastic samples",    cxxopts::value<size_t>()->default_value("5"))
      ("h,help", "Print usage");
  // clang-format on

  try {
    const auto result = options.parse(argc, argv);
    if (result.count("help") > 0) {
      std::cout << options.help() << "\n";
      std::exit(0);
    }
    options_out->lattice_size = result["lattice-size"].as<size_t>();
    options_out->kx = result["kx"].as<size_t>();
    options_out->ky = result["ky"].as<size_t>();
    options_out->kz = result["kz"].as<size_t>();
    options_out->t = result["hopping"].as<double>();
    options_out->U = result["interaction"].as<double>();
    options_out->beta = result["beta"].as<double>();
    options_out->dt = result["dt"].as<double>();
    options_out->krylov_steps = result["krylov-steps"].as<size_t>();
    options_out->steps = result["steps"].as<size_t>();
    options_out->num_samples = result["num-samples"].as<size_t>();
  } catch (const std::exception& ex) {
    std::cerr << "Argument error: " << ex.what() << "\n";
    std::cerr << options.help() << "\n";
    std::exit(1);
  }
}
}  // namespace

int main(int argc, char** argv) {
  CliOptions opts;
  parse_cli_options(argc, argv, &opts);

  if (opts.lattice_size == 0) {
    std::cerr << "Lattice size must be positive.\n";
    return 1;
  }
  if (opts.kx >= opts.lattice_size || opts.ky >= opts.lattice_size ||
      opts.kz >= opts.lattice_size) {
    std::cerr << "Total momentum components must be smaller than lattice size.\n";
    return 1;
  }
  if (opts.beta <= 0.0) {
    std::cerr << "Inverse temperature must be positive.\n";
    return 1;
  }
  if (opts.dt <= 0.0) {
    std::cerr << "Time step must be positive.\n";
    return 1;
  }
  if (opts.krylov_steps == 0) {
    std::cerr << "Krylov steps must be positive.\n";
    return 1;
  }
  if (opts.steps == 0) {
    std::cerr << "Steps must be positive.\n";
    return 1;
  }
  if (opts.num_samples == 0) {
    std::cerr << "Number of samples must be positive.\n";
    return 1;
  }

  const std::vector<size_t> lattice_size{opts.lattice_size, opts.lattice_size, opts.lattice_size};
  const std::vector<size_t> total_momentum{opts.kx, opts.ky, opts.kz};

  HubbardRelativeKinetic kinetic(lattice_size, total_momentum);
  HubbardRelativeInteraction onsite(lattice_size);
  const auto hamiltonian = opts.t * kinetic + opts.U * onsite;

  HubbardRelativeCurrent jx(lattice_size, total_momentum, opts.t, 0);

  std::vector<std::complex<double>> global_correlator(opts.steps, std::complex<double>(0.0, 0.0));

  std::cout << "Starting " << opts.num_samples << " simulations on " << omp_get_max_threads()
            << " threads...\n";

#pragma omp parallel
  {
    const unsigned int seed = static_cast<unsigned int>(std::time(nullptr)) ^
                              (static_cast<unsigned int>(omp_get_thread_num()) << 16);
    std::mt19937 rng(seed);

    std::vector<std::complex<double>> thread_sum(opts.steps, std::complex<double>(0.0, 0.0));

#pragma omp for schedule(static)
    for (int s = 0; s < static_cast<int>(opts.num_samples); ++s) {
      arma::cx_vec v_beta = random_complex_vector(hamiltonian.dimension(), rng);
      const double v_norm = arma::norm(v_beta);
      if (v_norm > 0.0) {
        v_beta /= v_norm;
      }

      EvolutionOptions evolve_options;
      evolve_options.krylov_steps = opts.krylov_steps;
      v_beta = imaginary_time_evolve_state(hamiltonian, v_beta, 0.5 * opts.beta, evolve_options);

      const double Z = std::real(arma::cdot(v_beta, v_beta));

      arma::cx_vec phi_1 = v_beta;
      arma::cx_vec phi_2 = jx.apply(v_beta);

      for (size_t i = 0; i < opts.steps; ++i) {
        thread_sum[i] += arma::cdot(phi_1, jx.apply(phi_2)) / Z;
        phi_1 = time_evolve_state(hamiltonian, phi_1, opts.dt, evolve_options);
        phi_2 = time_evolve_state(hamiltonian, phi_2, opts.dt, evolve_options);
      }

#pragma omp critical
      {
        std::cout << "Sample " << (s + 1) << "/" << opts.num_samples << " completed by thread "
                  << omp_get_thread_num() << "\n";
      }
    }

#pragma omp critical
    {
      for (size_t i = 0; i < opts.steps; ++i) {
        global_correlator[i] += thread_sum[i];
      }
    }
  }

  for (auto& c : global_correlator) {
    c /= static_cast<double>(opts.num_samples);
  }

  std::cout << std::setprecision(10);
  for (size_t i = 0; i < opts.steps; ++i) {
    std::cout << (static_cast<double>(i) * opts.dt) << " " << global_correlator[i].real() << " "
              << global_correlator[i].imag() << "\n";
  }
  return 0;
}
