#include <armadillo>
#include <complex>
#include <cstddef>
#include <exception>
#include <iostream>
#include <vector>

#include "cxxopts.hpp"
#include "hubbard_relative_operators.h"

struct CliOptions {
  size_t lattice_size = 16;
  size_t total_momentum = 0;
  double t = 1.0;
  double U = -4.0;
  size_t num_steps = 100;
};

void parse_cli_options(int argc, char** argv, CliOptions* options_out) {
  cxxopts::Options options("hubbard_relative_overlap",
                           "Compute overlap decay for the 3D Hubbard relative model");
  // clang-format off
  options.add_options()
      ("L,lattice-size", "Lattice size per dimension",  cxxopts::value<size_t>()->default_value("16"))
      ("P,total-momentum", "Total momentum value",      cxxopts::value<size_t>()->default_value("0"))
      ("t,hopping", "Hopping amplitude",                cxxopts::value<double>()->default_value("1.0"))
      ("U,interaction", "On-site interaction strength", cxxopts::value<double>()->default_value("-4.0"))
      ("n,num-steps", "Number of normalization steps",  cxxopts::value<size_t>()->default_value("100"))
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
    options_out->num_steps = result["num-steps"].as<size_t>();
  } catch (const std::exception& ex) {
    std::cerr << "Argument error: " << ex.what() << "\n";
    std::cerr << options.help() << "\n";
    std::exit(1);
  }
}

int main(int argc, char** argv) {
  CliOptions opts;
  parse_cli_options(argc, argv, &opts);

  const std::vector<size_t> lattice_size{opts.lattice_size, opts.lattice_size, opts.lattice_size};
  const std::vector<size_t> total_momentum{opts.total_momentum, opts.total_momentum,
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
