#pragma once

namespace brg {

/// Result of a single Block Renormalization Group step.
/// Used for both T=0 (where F=E) and finite-T calculations.
struct BrgStepResult {
  // Renormalized parameters
  double t_prime;
  double U_prime;
  double mu_prime;
  double K_prime;

  // Ground state energies (from diagonalization)
  double E1;  // (N=0, Sz=0)
  double E2;  // (N=2, Sz=0)
  double E3;  // (N=1, Sz=+1/2)
  double E4;  // (N=1, Sz=-1/2)

  // Free energies (F=E at T=0)
  double F1;
  double F2;
  double F3;
  double F4;

  // Lambda diagnostics
  double lambda_avg;        // average lambda amplitude
  double lambda_sq_avg;     // average lambda^2 (used for t')
  double lambda_spin_diff;  // max |lambda_up - lambda_down|
  double lambda_site_diff;  // max |lambda_i - lambda_j| across border sites
  double closure_error;     // max closure check deviation
};

/// Create a BrgStepResult for T=0 calculations where F=E.
inline BrgStepResult make_zero_t_result(double t_prime, double U_prime, double mu_prime,
                                        double K_prime, double E1, double E2, double E3, double E4,
                                        double lambda_avg, double lambda_spin_diff,
                                        double lambda_site_diff, double closure_error) {
  return BrgStepResult{
      t_prime,
      U_prime,
      mu_prime,
      K_prime,
      E1,
      E2,
      E3,
      E4,
      E1,  // F1 = E1 at T=0
      E2,  // F2 = E2 at T=0
      E3,  // F3 = E3 at T=0
      E4,  // F4 = E4 at T=0
      lambda_avg,
      lambda_avg * lambda_avg,  // lambda_sq_avg = lambda_avg^2 at T=0
      lambda_spin_diff,
      lambda_site_diff,
      closure_error,
  };
}

}  // namespace brg
