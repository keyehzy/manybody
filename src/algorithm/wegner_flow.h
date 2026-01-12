#pragma once

#include <armadillo>
#include <cstddef>
#include <functional>

#include "algorithm/dynamical_system.h"

arma::cx_mat wegner_flow(const arma::cx_mat& h0, double lmax, double dl,
                         IntegratorMethod method = IntegratorMethod::kRungeKutta4);

arma::cx_mat wegner_flow(const arma::cx_mat& h0, double lmax, double dl,
                         std::function<void(double, const arma::cx_mat&)> callback,
                         IntegratorMethod method = IntegratorMethod::kRungeKutta4);

arma::cx_mat block_wegner_flow(const arma::cx_mat& h0, size_t p_dim, double lmax, double dl,
                               IntegratorMethod method = IntegratorMethod::kRungeKutta4);

arma::cx_mat block_wegner_flow(const arma::cx_mat& h0, size_t p_dim, double lmax, double dl,
                               std::function<void(double, const arma::cx_mat&)> callback,
                               IntegratorMethod method = IntegratorMethod::kRungeKutta4);
