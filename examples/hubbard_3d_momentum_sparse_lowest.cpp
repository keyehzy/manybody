#include <armadillo>
#include <cstddef>
#include <exception>
#include <iostream>
#include <vector>

#include "algebra/basis.h"
#include "algebra/matrix_elements.h"
#include "algebra/model/hubbard_model_momentum.h"
#include "cxxopts.hpp"

struct CliOptions {
  size_t size_x = 2;
  size_t size_y = 2;
  size_t size_z = 2;
  size_t particles = 3;
  int spin_projection = 1;
  size_t kx = 0;
  size_t ky = 0;
  size_t kz = 0;
  double t = 1.0;
  double U = 4.0;
};

CliOptions parse_cli_options(int argc, char** argv) {
  CliOptions o;

  cxxopts::Options cli("hubbard_3d_momentum_sparse_lowest",
                       "Compute the lowest eigenvalue for the 3D Hubbard model in momentum space");
  // clang-format off
  cli.add_options()
      ("x,size-x", "Lattice size in x dimension",       cxxopts::value(o.size_x)->default_value("2"))
      ("y,size-y", "Lattice size in y dimension",       cxxopts::value(o.size_y)->default_value("2"))
      ("z,size-z", "Lattice size in z dimension",       cxxopts::value(o.size_z)->default_value("2"))
      ("N,particles", "Number of particles",            cxxopts::value(o.particles)->default_value("3"))
      ("S,spin", "Spin projection (n_up - n_down)",     cxxopts::value(o.spin_projection)->default_value("1"))
      ("kx", "Total momentum Kx component",             cxxopts::value(o.kx)->default_value("0"))
      ("ky", "Total momentum Ky component",             cxxopts::value(o.ky)->default_value("0"))
      ("kz", "Total momentum Kz component",             cxxopts::value(o.kz)->default_value("0"))
      ("t,hopping", "Hopping amplitude",                cxxopts::value(o.t)->default_value("1.0"))
      ("U,interaction", "On-site interaction strength", cxxopts::value(o.U)->default_value("4.0"))
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
  if (opts.size_x == 0 || opts.size_y == 0 || opts.size_z == 0) {
    std::cerr << "Lattice dimensions must be positive.\n";
    std::exit(1);
  }
  if (opts.particles == 0) {
    std::cerr << "Number of particles must be positive.\n";
    std::exit(1);
  }
  const size_t sites = opts.size_x * opts.size_y * opts.size_z;
  if (opts.particles > 2 * sites) {
    std::cerr << "Too many particles for the lattice size.\n";
    std::exit(1);
  }
  const int64_t particles_signed = static_cast<int64_t>(opts.particles);
  const int64_t spin = static_cast<int64_t>(opts.spin_projection);
  if (std::abs(spin) > particles_signed) {
    std::cerr << "Spin projection magnitude cannot exceed particle count.\n";
    std::exit(1);
  }
  if ((particles_signed + spin) % 2 != 0) {
    std::cerr << "Invalid spin projection: (particles + spin) must be even.\n";
    std::exit(1);
  }
  if (opts.kx >= opts.size_x || opts.ky >= opts.size_y || opts.kz >= opts.size_z) {
    std::cerr << "Momentum components must be less than corresponding lattice dimensions.\n";
    std::exit(1);
  }
}

int main(int argc, char** argv) {
  const CliOptions opts = parse_cli_options(argc, argv);
  validate_options(opts);

  const std::vector<size_t> size = {opts.size_x, opts.size_y, opts.size_z};
  const size_t sites = opts.size_x * opts.size_y * opts.size_z;
  const std::vector<size_t> total_momentum = {opts.kx, opts.ky, opts.kz};

  HubbardModelMomentum hubbard(opts.t, opts.U, size);
  Index index(size);
  Basis basis = Basis::with_fixed_particle_number_spin_momentum(
      sites, opts.particles, opts.spin_projection, index, total_momentum);

  const Expression hamiltonian = hubbard.hamiltonian();
  const Expression kinetic = hubbard.kinetic();
  const Expression interaction = hubbard.interaction();

  arma::sp_cx_mat H = compute_matrix_elements<arma::sp_cx_mat>(basis, hamiltonian);
  arma::sp_cx_mat K = compute_matrix_elements<arma::sp_cx_mat>(basis, kinetic);
  arma::sp_cx_mat U = compute_matrix_elements<arma::sp_cx_mat>(basis, interaction);

  arma::cx_vec eigenvalues;
  arma::cx_mat eigenvectors;
  if (!arma::eigs_gen(eigenvalues, eigenvectors, H, 1, "sr")) {
    std::cerr << "Sparse diagonalization failed." << std::endl;
    return 1;
  }

  const arma::cx_vec ground_state = eigenvectors.col(0);
  const std::complex<double> kinetic_expectation = arma::cdot(ground_state, K * ground_state);
  const std::complex<double> interaction_expectation = arma::cdot(ground_state, U * ground_state);

  std::cout << "3D Hubbard model in momentum space\n";
  std::cout << "L=(" << opts.size_x << "x" << opts.size_y << "x" << opts.size_z << ")"
            << ", N=" << opts.particles << ", Sz=" << opts.spin_projection << ", K=(" << opts.kx
            << "," << opts.ky << "," << opts.kz << ")\n";
  std::cout << "t=" << opts.t << ", U=" << opts.U << "\n";
  std::cout << "Basis size: " << basis.set.size() << "\n";
  std::cout << "Lowest eigenvalue (sparse): " << eigenvalues(0) << "\n";
  std::cout << "Kinetic energy <K>: " << kinetic_expectation << "\n";
  std::cout << "Interaction energy <U>: " << interaction_expectation << "\n";

  return 0;
}
