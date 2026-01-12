#include "algorithm/wegner_flow.h"

namespace {

arma::cx_mat diagonal_part(const arma::cx_mat& H) { return arma::diagmat(H.diag()); }

arma::cx_mat block_diagonal_part(const arma::cx_mat& H, size_t p_dim) {
  arma::cx_mat Hd(H.n_rows, H.n_cols, arma::fill::zeros);
  if (p_dim == 0 || p_dim >= H.n_rows) {
    Hd = H;
    return Hd;
  }

  Hd.submat(0, 0, p_dim - 1, p_dim - 1) = H.submat(0, 0, p_dim - 1, p_dim - 1);
  Hd.submat(p_dim, p_dim, H.n_rows - 1, H.n_cols - 1) =
      H.submat(p_dim, p_dim, H.n_rows - 1, H.n_cols - 1);
  return Hd;
}

struct WegnerFlowSystem {
  arma::cx_mat operator()(double /*l*/, const arma::cx_mat& H) const {
    const arma::cx_mat diag = diagonal_part(H);
    const arma::cx_mat eta = diag * H - H * diag;
    return eta * H - H * eta;
  }
};

struct BlockWegnerFlowSystem {
  explicit BlockWegnerFlowSystem(size_t p_dim) : p_dim(p_dim) {}

  arma::cx_mat operator()(double /*l*/, const arma::cx_mat& H) const {
    const arma::cx_mat Hd = block_diagonal_part(H, p_dim);
    const arma::cx_mat Hr = H - Hd;
    const arma::cx_mat eta = Hd * Hr - Hr * Hd;
    return eta * H - H * eta;
  }

  size_t p_dim;
};

}  // namespace

arma::cx_mat wegner_flow(const arma::cx_mat& h0, double lmax, double dl, IntegratorMethod method) {
  if (lmax <= 0.0 || dl <= 0.0) {
    return h0;
  }

  WegnerFlowSystem system;
  return integrate(system, h0, 0.0, lmax, dl, method);
}

arma::cx_mat block_wegner_flow(const arma::cx_mat& h0, size_t p_dim, double lmax, double dl,
                               IntegratorMethod method) {
  if (lmax <= 0.0 || dl <= 0.0 || p_dim == 0 || p_dim >= h0.n_rows) {
    return h0;
  }

  BlockWegnerFlowSystem system(p_dim);
  return integrate(system, h0, 0.0, lmax, dl, method);
}
