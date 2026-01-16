#include <complex>
#include <cstdlib>
#include <exception>
#include <iomanip>
#include <iostream>
#include <vector>

#include "algorithms/optical_conductivity.h"
#include "cxxopts.hpp"
#include "hubbard_relative_current_q_shared.h"

namespace {
CurrentCorrelationOptions parse_cli_options(int argc, char** argv) {
  CurrentCorrelationOptions o;

  cxxopts::Options cli(
      "hubbard_relative_optical_conductivity",
      "Compute optical conductivity from the q-dependent current-current correlator");
  // clang-format off
  cli.add_options()
      ("L,lattice-size", "Lattice size per dimension",  cxxopts::value(o.lattice_size)->default_value("8"))
      ("x,kx", "Total momentum Kx component",           cxxopts::value(o.kx)->default_value("0"))
      ("y,ky", "Total momentum Ky component",           cxxopts::value(o.ky)->default_value("0"))
      ("z,kz", "Total momentum Kz component",           cxxopts::value(o.kz)->default_value("0"))
      ("Qx,qx", "Transfer momentum qx component",       cxxopts::value(o.qx)->default_value("1"))
      ("Qy,qy", "Transfer momentum qy component",       cxxopts::value(o.qy)->default_value("0"))
      ("Qz,qz", "Transfer momentum qz component",       cxxopts::value(o.qz)->default_value("0"))
      ("D,direction", "Current operator direction",     cxxopts::value(o.direction)->default_value("0"))
      ("t,hopping", "Hopping amplitude",                cxxopts::value(o.t)->default_value("1.0"))
      ("U,interaction", "On-site interaction strength", cxxopts::value(o.U)->default_value("-10.0"))
      ("b,beta", "Inverse temperature",                 cxxopts::value(o.beta)->default_value("10.0"))
      ("d,dt", "Real-time step size",                   cxxopts::value(o.dt)->default_value("0.01"))
      ("k,krylov-steps", "Krylov subspace dimension",   cxxopts::value(o.krylov_steps)->default_value("20"))
      ("s,steps", "Real-time evolution steps",          cxxopts::value(o.steps)->default_value("1000"))
      ("n,num-samples", "Number of stochastic samples", cxxopts::value(o.num_samples)->default_value("5"))
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
}  // namespace

int main(int argc, char** argv) {
  const CurrentCorrelationOptions opts = parse_cli_options(argc, argv);
  opts.validate();

  const CurrentCorrelation correlator(opts);

  const std::vector<std::complex<double>> correlation =
      correlator.compute_current_current_correlation_q(&std::cerr);

  const double L = static_cast<double>(opts.lattice_size);
  const double volume = L * L * L;
  const OpticalConductivityResult result =
      compute_optical_conductivity(correlation, opts.dt, opts.beta, volume);

  std::cout << std::setprecision(10);
  for (size_t i = 0; i < result.frequencies.size(); ++i) {
    std::cout << result.frequencies[i] << " " << result.sigma[i].real() << " "
              << result.sigma[i].imag() << "\n";
  }

  std::cerr << "Optical Sum Rule Check\n";
  std::cerr << "Integral Re[sigma(w)] dw: " << result.sum_rule << "\n";
  return 0;
}
