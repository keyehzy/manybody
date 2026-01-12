#include "algorithm/wegner_flow.h"

namespace {

arma::cx_mat diagonal_part(const arma::cx_mat& H) { return arma::diagmat(H.diag()); }

struct WegnerFlowSystem {
  arma::cx_mat operator()(double /*l*/, const arma::cx_mat& H) const {
    const arma::cx_mat diag = diagonal_part(H);
    const arma::cx_mat eta = diag * H - H * diag;
    return eta * H - H * eta;
  }
};

}  // namespace

arma::cx_mat wegner_flow(const arma::cx_mat& h0, double lmax, double dl, IntegratorMethod method) {
  if (lmax <= 0.0 || dl <= 0.0) {
    return h0;
  }

  WegnerFlowSystem system;
  return integrate(system, h0, 0.0, lmax, dl, method);
}
