#pragma once

#include <complex>
#include <vector>

struct OpticalConductivityResult {
  std::vector<double> frequencies;
  std::vector<std::complex<double>> sigma;
  double sum_rule = 0.0;
};

OpticalConductivityResult compute_optical_conductivity(
    const std::vector<std::complex<double>>& correlation, double dt, double beta, double volume);
