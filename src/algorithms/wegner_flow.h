#pragma once

#include <armadillo>
#include <cstddef>
#include <functional>

#include "algorithms/dynamical_system.h"

arma::cx_mat wegner_flow(const arma::cx_mat& h0, double lmax, double dl,
                         IntegratorMethod method = IntegratorMethod::kRungeKutta4);

arma::cx_mat wegner_flow(const arma::cx_mat& h0, double lmax, double dl,
                         std::function<void(double, const arma::cx_mat&)> callback,
                         IntegratorMethod method = IntegratorMethod::kRungeKutta4);

/// Block Wegner flow that decouples the leading p_dim x p_dim block (P space) from the rest.
/// If p_dim == 0 or p_dim >= h0.n_rows there is no off-block part, so h0 is returned unchanged
/// (and the callback, if any, is never invoked). The same holds for lmax <= 0 or dl <= 0.
arma::cx_mat block_wegner_flow(const arma::cx_mat& h0, size_t p_dim, double lmax, double dl,
                               IntegratorMethod method = IntegratorMethod::kRungeKutta4);

arma::cx_mat block_wegner_flow(const arma::cx_mat& h0, size_t p_dim, double lmax, double dl,
                               std::function<void(double, const arma::cx_mat&)> callback,
                               IntegratorMethod method = IntegratorMethod::kRungeKutta4);
