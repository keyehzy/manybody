#pragma once

#include <armadillo>
#include <cmath>
#include <limits>

namespace brg {

/// Minimum temperature threshold below which we use T=0 approximation.
constexpr double kMinTemperature = 1e-15;

/// Maximum beta (inverse temperature) to avoid numerical overflow.
constexpr double kMaxBeta = 1e15;

/// Check if the given temperature should be treated as T=0.
inline bool use_zero_temperature(double T) {
  if (T <= 0.0) {
    return true;
  }
  if (T < kMinTemperature) {
    return true;
  }
  const double beta = 1.0 / T;
  return beta > kMaxBeta;
}

/// Thermal weights and associated thermodynamic quantities for a set of energy levels.
struct ThermalWeights {
  arma::vec weights;         // normalized Boltzmann weights: exp(-beta E_n) / Z
  double logZ = 0.0;         // log of partition function
  double free_energy = 0.0;  // Helmholtz free energy: F = -T log Z
};

/// Compute thermal (Boltzmann) weights for a set of eigenvalues at inverse temperature beta.
/// Uses numerically stable shifted exponentials to avoid overflow.
inline ThermalWeights compute_thermal_weights(const arma::vec& evals, double beta) {
  ThermalWeights w;
  const size_t n = evals.n_elem;
  w.weights.set_size(n);
  w.weights.zeros();

  if (n == 0) {
    w.logZ = -std::numeric_limits<double>::infinity();
    w.free_energy = 0.0;
    return w;
  }

  // Shift energies by ground state to avoid overflow
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

}  // namespace brg
