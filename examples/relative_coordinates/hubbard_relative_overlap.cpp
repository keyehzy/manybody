#include <armadillo>
#include <complex>
#include <cstddef>
#include <exception>
#include <iostream>
#include <vector>

#include "cxxopts.hpp"
#include "numerics/hubbard_relative_operators.h"

struct CliOptions {
  size_t lattice_size = 16;
  int64_t total_momentum = 0;
  double t = 1.0;
  double U = -4.0;
  size_t num_steps = 100;
};

CliOptions parse_cli_options(int argc, char** argv) {
  CliOptions o;

  cxxopts::Options cli("hubbard_relative_overlap",
                       "Compute overlap decay for the 3D Hubbard relative model");
  // clang-format off
  cli.add_options()
      ("L,lattice-size", "Lattice size per dimension",  cxxopts::value(o.lattice_size)->default_value("16"))
      ("P,total-momentum", "Total momentum value",      cxxopts::value(o.total_momentum)->default_value("0"))
      ("t,hopping", "Hopping amplitude",                cxxopts::value(o.t)->default_value("1.0"))
      ("U,interaction", "On-site interaction strength", cxxopts::value(o.U)->default_value("-4.0"))
      ("n,num-steps", "Number of normalization steps",  cxxopts::value(o.num_steps)->default_value("100"))
      ("h,help", "Print usage");
  // clang-format on

  try {
    auto result = cli.parse(argc, argv);
    if (result.count("help")) {
      std::cout << cli.help() << "\n";
      std::exit(0);
    }
  } catch (const std::exception& e) {
    std::cerr << "Argument error: " << e.what() << "\n" << cli.help() << "\n";
    std::exit(1);
  }

  return o;
}

int main(int argc, char** argv) {
  const CliOptions opts = parse_cli_options(argc, argv);

  const std::vector<size_t> lattice_size{opts.lattice_size, opts.lattice_size, opts.lattice_size};
  const std::vector<int64_t> total_momentum{opts.total_momentum, opts.total_momentum,
                                            opts.total_momentum};

  HubbardRelative hamiltonian(lattice_size, total_momentum, opts.t, opts.U);

  arma::cx_vec v0(hamiltonian.dimension(), arma::fill::zeros);
  v0(0) = 1.0;

  arma::cx_vec state = v0;
  for (size_t i = 0; i <= opts.num_steps; ++i) {
    const double c_i = std::norm(arma::cdot(v0, state));
    std::cout << i << " " << c_i << "\n";
    state = arma::normalise(hamiltonian.apply(state));
  }
  return 0;
}
