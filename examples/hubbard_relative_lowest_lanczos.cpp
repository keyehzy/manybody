#include <algorithm>
#include <armadillo>
#include <cstddef>
#include <exception>
#include <iostream>
#include <vector>

#include "cxxopts.hpp"
#include "hubbard_relative_operators.h"
#include "numerics/lanczos.h"

struct CliOptions {
  size_t lattice_size = 16;
  size_t total_momentum = 0;
  double t = 1.0;
  double U = -4.0;
  size_t lanczos_steps = 50;
};

void parse_cli_options(int argc, char** argv, CliOptions* options_out) {
  cxxopts::Options options("hubbard_relative_lowest_lanczos",
                           "Compute the lowest eigenvalue for the 3D relative Hubbard model");
  // clang-format off
  options.add_options()
      ("L,lattice-size", "Lattice size per dimension",  cxxopts::value<size_t>()->default_value("16"))
      ("P,total-momentum", "Total momentum value",      cxxopts::value<size_t>()->default_value("0"))
      ("t,hopping", "Hopping amplitude",                cxxopts::value<double>()->default_value("1.0"))
      ("U,interaction", "On-site interaction strength", cxxopts::value<double>()->default_value("-4.0"))
      ("k,lanczos-steps", "Lanczos iteration steps",    cxxopts::value<size_t>()->default_value("50"))
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
    options_out->lanczos_steps = result["lanczos-steps"].as<size_t>();
  } catch (const std::exception& ex) {
    std::cerr << "Argument error: " << ex.what() << "\n";
    std::cerr << options.help() << "\n";
    std::exit(1);
  }
}

int main(int argc, char** argv) {
  CliOptions opts;
  parse_cli_options(argc, argv, &opts);

  if (opts.lattice_size == 0) {
    std::cerr << "Lattice size must be positive.\n";
    return 1;
  }
  if (opts.total_momentum >= opts.lattice_size) {
    std::cerr << "Total momentum must be smaller than lattice size.\n";
    return 1;
  }
  if (opts.lanczos_steps == 0) {
    std::cerr << "Lanczos steps must be positive.\n";
    return 1;
  }

  const std::vector<size_t> lattice_size{opts.lattice_size, opts.lattice_size, opts.lattice_size};
  const std::vector<size_t> total_momentum{opts.total_momentum, opts.total_momentum,
                                           opts.total_momentum};

  HubbardRelativeKinetic kinetic(lattice_size, total_momentum);
  HubbardRelativeInteraction onsite(lattice_size);

  auto hamiltonian = opts.t * kinetic + opts.U * onsite;

  const size_t dimension = hamiltonian.dimension();
  const size_t lanczos_steps = std::min(opts.lanczos_steps, dimension);
  const auto eigenpair = find_min_eigenpair(hamiltonian, lanczos_steps);

  const arma::vec residual =
      hamiltonian.apply(eigenpair.vector) - eigenpair.value * eigenpair.vector;
  const double residual_norm = arma::norm(residual);

  std::cout << "Relative-coordinate Hubbard Hamiltonian (3D)\n";
  std::cout << "L=" << opts.lattice_size << ", K=" << opts.total_momentum << ", t=" << opts.t
            << ", U=" << opts.U << "\n";
  std::cout << "dimension=" << dimension << ", lanczos_steps=" << lanczos_steps << "\n";
  std::cout << "lowest_eigenvalue=" << eigenpair.value << "\n";
  std::cout << "residual_norm=" << residual_norm << "\n";
  return 0;
}
