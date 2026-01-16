#include "hubbard_relative_current_q_shared.h"

#include <omp.h>

#include <cstdlib>
#include <ctime>
#include <iostream>
#include <random>

#include "hubbard_relative_operators.h"
#include "numerics/evolve_state.h"
#include "numerics/linear_operator.h"

void CurrentCorrelationOptions::validate() const {
  if (lattice_size == 0) {
    std::cerr << "Lattice size must be positive.\n";
    std::exit(1);
  }
  if (beta <= 0.0) {
    std::cerr << "Inverse temperature must be positive.\n";
    std::exit(1);
  }
  if (dt <= 0.0) {
    std::cerr << "Time step must be positive.\n";
    std::exit(1);
  }
  if (krylov_steps == 0) {
    std::cerr << "Krylov steps must be positive.\n";
    std::exit(1);
  }
  if (steps == 0) {
    std::cerr << "Steps must be positive.\n";
    std::exit(1);
  }
  if (num_samples == 0) {
    std::cerr << "Number of samples must be positive.\n";
    std::exit(1);
  }
  constexpr size_t dims = 3;
  if (direction >= dims) {
    std::cerr << "Current direction must be in [0, 2].\n";
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

std::vector<std::complex<double>> CurrentCorrelation::compute_current_current_correlation_q(
    std::ostream* log_stream) const {
  const std::vector<size_t> lattice_size{options_.lattice_size, options_.lattice_size,
                                         options_.lattice_size};
  const std::vector<int64_t> total_momentum{options_.kx, options_.ky, options_.kz};
  const std::vector<int64_t> transfer_momentum{options_.qx, options_.qy, options_.qz};
  const std::vector<int64_t> total_momentum_q{options_.kx + options_.qx, options_.ky + options_.qy,
                                              options_.kz + options_.qz};
  const std::vector<int64_t> transfer_momentum_neg{-options_.qx, -options_.qy, -options_.qz};

  const HubbardRelative hamiltonian(lattice_size, total_momentum, options_.t, options_.U);
  const HubbardRelative hamiltonian_q(lattice_size, total_momentum_q, options_.t, options_.U);

  const CurrentRelative_Q j_plus(lattice_size, options_.t, total_momentum, transfer_momentum,
                                 options_.direction);
  const CurrentRelative_Q j_minus(lattice_size, options_.t, total_momentum_q, transfer_momentum_neg,
                                  options_.direction);

  std::vector<std::complex<double>> global_correlator(options_.steps,
                                                      std::complex<double>(0.0, 0.0));

  if (log_stream) {
    (*log_stream) << "Starting " << options_.num_samples << " simulations on "
                  << omp_get_max_threads() << " threads...\n";
  }

#pragma omp parallel
  {
    const unsigned int seed = static_cast<unsigned int>(std::time(nullptr)) ^
                              (static_cast<unsigned int>(omp_get_thread_num()) << 16);
    std::mt19937 rng(seed);

    std::vector<std::complex<double>> thread_sum(options_.steps, std::complex<double>(0.0, 0.0));

#pragma omp for schedule(static)
    for (int s = 0; s < static_cast<int>(options_.num_samples); ++s) {
      arma::cx_vec v_beta = random_complex_vector(hamiltonian.dimension(), rng);
      const double v_norm = arma::norm(v_beta);
      if (v_norm > 0.0) {
        v_beta /= v_norm;
      }

      EvolutionOptions evolve_options;
      evolve_options.krylov_steps = options_.krylov_steps;
      v_beta =
          imaginary_time_evolve_state(hamiltonian, v_beta, 0.5 * options_.beta, evolve_options);

      const double Z = std::real(arma::cdot(v_beta, v_beta));

      arma::cx_vec phi_1 = v_beta;
      arma::cx_vec phi_2 = j_plus.apply(v_beta);

      for (size_t i = 0; i < options_.steps; ++i) {
        thread_sum[i] += arma::cdot(phi_1, j_minus.apply(phi_2)) / Z;
        phi_1 = time_evolve_state(hamiltonian, phi_1, options_.dt, evolve_options);
        phi_2 = time_evolve_state(hamiltonian_q, phi_2, options_.dt, evolve_options);
      }

#pragma omp critical
      {
        if (log_stream) {
          (*log_stream) << "Sample " << (s + 1) << "/" << options_.num_samples
                        << " completed by thread " << omp_get_thread_num() << "\n";
        }
      }
    }

#pragma omp critical
    {
      for (size_t i = 0; i < options_.steps; ++i) {
        global_correlator[i] += thread_sum[i];
      }
    }
  }

  for (auto& c : global_correlator) {
    c /= static_cast<double>(options_.num_samples);
  }

  return global_correlator;
}
