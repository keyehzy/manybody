#include "algorithms/optical_conductivity.h"

#include <cmath>
#include <cstddef>
#include <numbers>
#include <stdexcept>

OpticalConductivityResult compute_optical_conductivity(
    const std::vector<std::complex<double>>& correlation, double dt, double beta, double volume) {
  const std::size_t n_samples = correlation.size();
  if (n_samples < 2) {
    throw std::invalid_argument("correlation must have at least 2 samples");
  }
  if (!(dt > 0.0) || !(volume > 0.0) || !(beta >= 0.0)) {
    throw std::invalid_argument("dt>0, volume>0, beta>=0 required");
  }

  const double total_time = (static_cast<double>(n_samples) - 1.0) * dt;
  if (total_time <= 0.0) {
    throw std::invalid_argument("total time window must be positive");
  }

  const double w_min = 0.0;
  const double w_max = std::numbers::pi / dt;

  const double eta = 4.0 / total_time;
  const double dw = eta / 4.0;

  const std::size_t n_freqs = static_cast<std::size_t>(std::floor((w_max - w_min) / dw)) + 1;

  std::vector<std::complex<double>> series(n_samples);

  const double damp_step = std::exp(-eta * dt);
  double damp = 1.0;

  for (std::size_t k = 0; k < n_samples; ++k) {
    double weight = dt;
    if (k == 0 || k == n_samples - 1) {
      weight *= 0.5;
    }

    series[k] = correlation[k] * (weight * damp);
    damp *= damp_step;
  }

  OpticalConductivityResult result;
  result.frequencies.reserve(n_freqs);
  result.sigma.reserve(n_freqs);

  double sum_rule = 0.0;

  for (std::size_t i = 0; i < n_freqs; ++i) {
    const double omega = w_min + static_cast<double>(i) * dw;
    result.frequencies.push_back(omega);

    const std::complex<double> phase_step = std::exp(std::complex<double>(0.0, omega * dt));

    std::complex<double> phase(1.0, 0.0);
    std::complex<double> integral(0.0, 0.0);

    for (std::size_t k = 0; k < n_samples; ++k) {
      integral += series[k] * phase;
      phase *= phase_step;
    }

    double prefactor = 0.0;
    if (std::abs(omega) < 1e-12) {
      prefactor = beta / volume;
    } else {
      prefactor = (-std::expm1(-beta * omega)) / (omega * volume);
    }

    const std::complex<double> sigma = prefactor * integral;
    result.sigma.push_back(sigma);

    const double weight = (i == 0 || i == n_freqs - 1) ? 0.5 : 1.0;
    sum_rule += sigma.real() * weight;
  }

  result.sum_rule = sum_rule * dw;
  return result;
}
