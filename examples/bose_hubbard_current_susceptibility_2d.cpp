#include <armadillo>
#include <cstddef>
#include <cstdlib>
#include <exception>
#include <iomanip>
#include <iostream>

#include "algebra/boson/basis.h"
#include "algebra/boson/model/hubbard_model.h"
#include "algebra/matrix_elements.h"
#include "algorithms/static_susceptibility.h"
#include "cxxopts.hpp"

struct CliOptions {
  size_t size_x = 3;
  size_t size_y = 3;
  size_t particles = 3;
  size_t direction = 0;
  size_t u_steps = 11;
  double t = 1.0;
  double u_min = 0.0;
  double u_max = 10.0;
  double gap_tolerance = 1e-10;
};

CliOptions parse_cli_options(int argc, char** argv) {
  CliOptions o;

  cxxopts::Options cli("bose_hubbard_current_susceptibility_2d",
                       "Scan the zero-temperature integrated current-current correlator for the "
                       "2D Bose-Hubbard model");
  // clang-format off
  cli.add_options()
      ("x,size-x", "Lattice size in x dimension",             cxxopts::value(o.size_x)->default_value("3"))
      ("y,size-y", "Lattice size in y dimension",             cxxopts::value(o.size_y)->default_value("3"))
      ("N,particles", "Number of bosons",                     cxxopts::value(o.particles)->default_value("3"))
      ("D,direction", "Current operator direction (0=x, 1=y)", cxxopts::value(o.direction)->default_value("0"))
      ("t,hopping", "Hopping amplitude",                      cxxopts::value(o.t)->default_value("1.0"))
      ("u-min", "Minimum on-site interaction U",              cxxopts::value(o.u_min)->default_value("0.0"))
      ("u-max", "Maximum on-site interaction U",              cxxopts::value(o.u_max)->default_value("10.0"))
      ("u-steps", "Number of U points",                       cxxopts::value(o.u_steps)->default_value("11"))
      ("gap-tolerance", "Degenerate-state gap tolerance",     cxxopts::value(o.gap_tolerance)->default_value("1e-10"))
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
  if (opts.size_x == 0 || opts.size_y == 0) {
    std::cerr << "Lattice dimensions must be positive.\n";
    std::exit(1);
  }
  if (opts.particles == 0) {
    std::cerr << "Number of particles must be positive.\n";
    std::exit(1);
  }
  if (opts.direction >= 2) {
    std::cerr << "Direction must be 0 or 1.\n";
    std::exit(1);
  }
  if (opts.u_steps == 0) {
    std::cerr << "u-steps must be positive.\n";
    std::exit(1);
  }
  if (opts.gap_tolerance < 0.0) {
    std::cerr << "gap-tolerance must be non-negative.\n";
    std::exit(1);
  }
  const size_t sites = opts.size_x * opts.size_y;
  if (opts.particles > BosonBasis::key_type::max_size()) {
    std::cerr << "Too many particles for the boson term storage size.\n";
    std::exit(1);
  }
  if (sites - 1 > BosonOperator::max_index()) {
    std::cerr << "Too many sites for the operator index storage size.\n";
    std::exit(1);
  }
}

double u_value(const CliOptions& opts, size_t step) {
  if (opts.u_steps == 1) {
    return opts.u_min;
  }

  const double fraction = static_cast<double>(step) / static_cast<double>(opts.u_steps - 1);
  return opts.u_min + fraction * (opts.u_max - opts.u_min);
}

int main(int argc, char** argv) {
  const CliOptions opts = parse_cli_options(argc, argv);
  validate_options(opts);

  const BoseHubbardModel2D base_model(opts.t, 1.0, opts.size_x, opts.size_y);
  const BosonBasis basis = BosonBasis::with_fixed_particle_number_and_spin(
      base_model.num_sites, opts.particles, static_cast<int>(opts.particles));

  const arma::cx_mat kinetic = compute_matrix_elements<arma::cx_mat>(basis, base_model.kinetic());
  const arma::cx_mat interaction =
      compute_matrix_elements<arma::cx_mat>(basis, base_model.interaction());
  const arma::cx_mat current =
      compute_matrix_elements<arma::cx_mat>(basis, base_model.current(opts.direction));

  std::cout << std::setprecision(12);
  std::cout << "# 2D Bose-Hubbard zero-temperature static current susceptibility\n";
  std::cout << "# L=" << opts.size_x << "x" << opts.size_y << " N=" << opts.particles
            << " basis_size=" << basis.set.size() << " t=" << opts.t
            << " direction=" << opts.direction << "\n";
  std::cout << "# U ground_energy susceptibility susceptibility_per_site skipped_states\n";

  const double area = static_cast<double>(base_model.num_sites);
  for (size_t step = 0; step < opts.u_steps; ++step) {
    const double U = u_value(opts, step);
    const arma::cx_mat hamiltonian = kinetic + U * interaction;

    arma::vec eigenvalues;
    arma::cx_mat eigenvectors;
    if (!arma::eig_sym(eigenvalues, eigenvectors, hamiltonian)) {
      std::cerr << "Diagonalization failed at U=" << U << ".\n";
      return 1;
    }

    const StaticSusceptibilityResult susceptibility =
        compute_zero_temperature_static_susceptibility(eigenvalues, eigenvectors, current,
                                                       opts.gap_tolerance);

    std::cout << U << " " << susceptibility.ground_energy << " " << susceptibility.value << " "
              << susceptibility.value / area << " " << susceptibility.skipped_states << "\n";
  }

  return 0;
}
