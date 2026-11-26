#include <algorithm>
#include <armadillo>
#include <cstddef>
#include <exception>
#include <iostream>
#include <vector>

#include "cxxopts.hpp"
#include "numerics/hubbard_relative_operators.h"
#include "numerics/lanczos.h"

struct CliOptions {
  size_t lattice_size = 16;
  int64_t total_momentum = 0;
  double t = 1.0;
  double U = -4.0;
  size_t lanczos_steps = 50;
};

CliOptions parse_cli_options(int argc, char** argv) {
  CliOptions o;

  cxxopts::Options cli("hubbard_relative_lowest_lanczos",
                       "Compute the lowest eigenvalue for the 3D relative Hubbard model");
  // clang-format off
  cli.add_options()
      ("L,lattice-size", "Lattice size per dimension",  cxxopts::value(o.lattice_size)->default_value("16"))
      ("P,total-momentum", "Total momentum value",      cxxopts::value(o.total_momentum)->default_value("0"))
      ("t,hopping", "Hopping amplitude",                cxxopts::value(o.t)->default_value("1.0"))
      ("U,interaction", "On-site interaction strength", cxxopts::value(o.U)->default_value("-4.0"))
      ("k,lanczos-steps", "Lanczos iteration steps",    cxxopts::value(o.lanczos_steps)->default_value("50"))
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

void validate_options(const CliOptions& opts) {
  if (opts.lattice_size == 0) {
    std::cerr << "Lattice size must be positive.\n";
    std::exit(1);
  }
  if (opts.lanczos_steps == 0) {
    std::cerr << "Lanczos steps must be positive.\n";
    std::exit(1);
  }
}

int main(int argc, char** argv) {
  const CliOptions opts = parse_cli_options(argc, argv);
  validate_options(opts);

  const std::vector<size_t> lattice_size{opts.lattice_size, opts.lattice_size, opts.lattice_size};
  const std::vector<int64_t> total_momentum{opts.total_momentum, opts.total_momentum,
                                            opts.total_momentum};

  HubbardRelative hamiltonian(lattice_size, total_momentum, opts.t, opts.U);

  const size_t dimension = hamiltonian.dimension();
  const size_t lanczos_steps = std::min(opts.lanczos_steps, dimension);
  const auto eigenpair = find_min_eigenpair(hamiltonian, lanczos_steps);

  const arma::cx_vec residual =
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
