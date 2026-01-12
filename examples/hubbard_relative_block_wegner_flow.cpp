#include <armadillo>
#include <cmath>
#include <cstddef>
#include <exception>
#include <iostream>

#include "algorithm/wegner_flow.h"
#include "cxxopts.hpp"
#include "hubbard_relative_operators.h"

struct CliOptions {
  size_t lattice_size = 6;
  size_t total_momentum = 0;
  double t = 1.0;
  double U = -4.0;
  double lmax = 5.0;
  double dl = 0.01;
};

void parse_cli_options(int argc, char** argv, CliOptions* options_out) {
  cxxopts::Options options("hubbard_relative_block_wegner_flow",
                           "Track block Wegner flow for the 3D relative Hubbard model");
  // clang-format off
  options.add_options()
      ("L,lattice-size", "Lattice size per dimension",  cxxopts::value<size_t>()->default_value("6"))
      ("P,total-momentum", "Total momentum value",      cxxopts::value<size_t>()->default_value("0"))
      ("t,hopping", "Hopping amplitude",                cxxopts::value<double>()->default_value("1.0"))
      ("U,interaction", "On-site interaction strength", cxxopts::value<double>()->default_value("-4.0"))
      ("l,lmax", "Maximum flow parameter",              cxxopts::value<double>()->default_value("5.0"))
      ("d,dl", "Flow step size",                         cxxopts::value<double>()->default_value("0.01"))
      ("h,help", "Print usage");
  // clang-format on

  try {
    const auto result = options.parse(argc, argv);
    if (result.count("help") > 0) {
      std::cout << options.help() << "\n";
      std::exit(0);
    }
    options_out->lattice_size = result["lattice-size"].as<size_t>();
    options_out->total_momentum = result["total-momentum"].as<size_t>();
    options_out->t = result["hopping"].as<double>();
    options_out->U = result["interaction"].as<double>();
    options_out->lmax = result["lmax"].as<double>();
    options_out->dl = result["dl"].as<double>();
  } catch (const std::exception& ex) {
    std::cerr << "Argument error: " << ex.what() << "\n";
    std::cerr << options.help() << "\n";
    std::exit(1);
  }
}

// Measures the magnitude of the off-block-diagonal elements
// of a complex matrix H, assuming a block partition at index p_dim.
//
// The matrix is conceptually divided as:
//
//     [ A  B ]
//     [ C  D ]
//
// where:
//   - A is a p_dim x p_dim block,
//   - B is the upper off-diagonal block (rows 0..p_dim-1, cols p_dim..end),
//   - C is the lower off-diagonal block (rows p_dim..end, cols 0..p_dim-1),
//   - D is the remaining block.
//
// The function computes:
//
//     sqrt( ||B||_F^2 + ||C||_F^2 )
//
// where ||·||_F denotes the Frobenius norm.
double off_block_norm(const arma::cx_mat& H, size_t p_dim) {
  if (p_dim == 0 || p_dim >= H.n_rows) {
    return 0.0;
  }

  const arma::cx_mat upper = H.submat(0, p_dim, p_dim - 1, H.n_cols - 1);
  const arma::cx_mat lower = H.submat(p_dim, 0, H.n_rows - 1, p_dim - 1);
  const double upper_norm = arma::norm(upper, "fro");
  const double lower_norm = arma::norm(lower, "fro");
  return std::sqrt(upper_norm * upper_norm + lower_norm * lower_norm);
}

// Builds the explicit matrix form of a linear operator by
// applying it to each vector of the canonical basis in R^dimension.
// The resulting matrix H satisfies:
//
//     H.col(j) = hamiltonian * e_j
//
// where e_j is the j-th standard basis vector.
arma::cx_mat build_hamiltonian_matrix(const LinearOperator<arma::vec>& hamiltonian,
                                      size_t dimension) {
  arma::mat h_real(dimension, dimension, arma::fill::zeros);
  arma::vec basis_vector(dimension, arma::fill::zeros);
  for (size_t j = 0; j < dimension; ++j) {
    basis_vector.zeros();
    basis_vector(j) = 1.0;
    const arma::vec column = hamiltonian.apply(basis_vector);
    h_real.col(j) = column;
  }
  return arma::cx_mat(h_real, arma::mat(dimension, dimension, arma::fill::zeros));
}

int main(int argc, char** argv) {
  CliOptions opts;
  parse_cli_options(argc, argv, &opts);

  const size_t total_size = opts.lattice_size * opts.lattice_size * opts.lattice_size;
  constexpr size_t kBlockDim = 1;

  HubbardRelativeKinetic3D kinetic(opts.lattice_size, opts.total_momentum, opts.total_momentum,
                                   opts.total_momentum);
  HubbardRelativeInteraction onsite(total_size);
  auto hamiltonian = opts.t * kinetic + opts.U * onsite;
  const arma::cx_mat h0 = build_hamiltonian_matrix(hamiltonian, total_size);

  auto callback = [](double l, const arma::cx_mat& H) {
    std::cout << l << " " << (off_block_norm(H, kBlockDim) / H(0, 0)).real() << "\n";
  };

  block_wegner_flow(h0, kBlockDim, opts.lmax, opts.dl, callback, IntegratorMethod::kRungeKutta4);
  return 0;
}
