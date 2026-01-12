#pragma once

#include <armadillo>

#include "algorithm/dynamical_system.h"

arma::cx_mat wegner_flow(const arma::cx_mat& h0, double lmax, double dl,
                         IntegratorMethod method = IntegratorMethod::kRungeKutta4);
