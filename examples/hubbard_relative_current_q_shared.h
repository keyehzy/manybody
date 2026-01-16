#pragma once

#include <complex>
#include <cstddef>
#include <cstdint>
#include <ostream>
#include <vector>

struct CurrentCorrelationOptions {
  size_t lattice_size = 8;
  int64_t kx = 0;
  int64_t ky = 0;
  int64_t kz = 0;
  int64_t qx = 1;
  int64_t qy = 0;
  int64_t qz = 0;
  size_t direction = 0;
  double t = 1.0;
  double U = -10.0;
  double beta = 10.0;
  double dt = 0.01;
  size_t krylov_steps = 20;
  size_t steps = 1000;
  size_t num_samples = 5;

  void validate() const;
};

class CurrentCorrelation {
 public:
  explicit CurrentCorrelation(CurrentCorrelationOptions options) : options_(options) {}

  const CurrentCorrelationOptions& options() const { return options_; }

 std::vector<std::complex<double>> compute_current_current_correlation_q(
      std::ostream* log_stream = nullptr) const;

 private:
  CurrentCorrelationOptions options_;
};
