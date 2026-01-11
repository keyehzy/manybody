#include <armadillo>
#include <cstddef>
#include <exception>
#include <iostream>

#include "cxxopts.hpp"
#include "hubbard_relative_operators.h"

int main(int argc, char** argv) {
  cxxopts::Options options("hubbard_relative_overlap",
                           "Compute overlap decay for the 3D Hubbard relative model");
  options.add_options()("L,lattice-size", "Lattice size per dimension",
                        cxxopts::value<size_t>()->default_value("16"))(
      "P,total-momentum", "Total momentum value", cxxopts::value<size_t>()->default_value("0"))(
      "t,hopping", "Hopping amplitude", cxxopts::value<double>()->default_value("1.0"))(
      "U,interaction", "On-site interaction strength",
      cxxopts::value<double>()->default_value("-4.0"))(
      "n,num-steps", "Number of normalization steps",
      cxxopts::value<size_t>()->default_value("100"))("h,help", "Print usage");

  size_t lattice_size = 16;
  size_t total_momentum = 0;
  double t = 1.0;
  double U = -4.0;
  size_t num_steps = 100;
  try {
    const auto result = options.parse(argc, argv);
    if (result.count("help") > 0) {
      std::cout << options.help() << "\n";
      return 0;
    }
    lattice_size = result["lattice-size"].as<size_t>();
    total_momentum = result["total-momentum"].as<size_t>();
    t = result["hopping"].as<double>();
    U = result["interaction"].as<double>();
    num_steps = result["num-steps"].as<size_t>();
  } catch (const std::exception& ex) {
    std::cerr << "Argument error: " << ex.what() << "\n";
    std::cerr << options.help() << "\n";
    return 1;
  }

  const size_t total_size = lattice_size * lattice_size * lattice_size;

  HubbardRelativeKinetic3D kinetic(lattice_size, total_momentum, total_momentum, total_momentum);
  HubbardRelativeInteraction onsite(total_size);

  auto hamiltonian = t * kinetic + U * onsite;

  arma::vec v0(total_size, arma::fill::zeros);
  v0(0) = 1.0;

  arma::vec state = v0;
  for (size_t i = 0; i <= num_steps; ++i) {
    const double c_i = std::pow(arma::dot(v0, state), 2);
    std::cout << i << " " << c_i << "\n";
    state = arma::normalise(hamiltonian.apply(state));
  }
  return 0;
}
