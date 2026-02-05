#include <armadillo>
#include <cstddef>
#include <exception>
#include <iostream>
#include <vector>

#include "algebra/fermion/basis.h"
#include "algebra/matrix_elements.h"
#include "algebra/model/hubbard_model_momentum.h"
#include "cxxopts.hpp"
#include "numerics/lanczos.h"
#include "numerics/linear_operator.h"

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
  size_t lanczos_steps = 50;
};

CliOptions parse_cli_options(int argc, char** argv) {
  CliOptions o;

  cxxopts::Options cli("hubbard_momentum_sparse_lanczos",
                       "Compute ground state of 3D Hubbard model using sparse matrix Lanczos");
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
  if (opts.size_x == 0 || opts.size_y == 0 || opts.size_z == 0) {
    std::cerr << "Lattice dimensions must be positive.\n";
    std::exit(1);
  }
  if (opts.particles == 0) {
    std::cerr << "Number of particles must be positive.\n";
    std::exit(1);
  }
  if (opts.lanczos_steps == 0) {
    std::cerr << "Lanczos steps must be positive.\n";
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

  if (basis.set.empty()) {
    std::cerr << "No basis states for the chosen particle number, spin, and momentum.\n";
    return 1;
  }

  const Expression hamiltonian = hubbard.hamiltonian();
  arma::sp_cx_mat H = compute_matrix_elements<arma::sp_cx_mat>(basis, hamiltonian);

  SparseComplexMatrixOperator op(std::move(H));
  const size_t dimension = op.dimension();
  const size_t lanczos_steps = std::min(opts.lanczos_steps, dimension);

  const auto eigenpair = find_min_eigenpair(op, lanczos_steps);
  const arma::cx_vec residual = op.apply(eigenpair.vector) - eigenpair.value * eigenpair.vector;
  const double residual_norm = arma::norm(residual);

  std::cout << "3D Hubbard model in momentum space (sparse matrix Lanczos)\n";
  std::cout << "L=(" << opts.size_x << "x" << opts.size_y << "x" << opts.size_z << ")"
            << ", N=" << opts.particles << ", Sz=" << opts.spin_projection << ", K=(" << opts.kx
            << "," << opts.ky << "," << opts.kz << ")\n";
  std::cout << "t=" << opts.t << ", U=" << opts.U << "\n";
  std::cout << "Basis size: " << basis.set.size() << "\n";
  std::cout << "Lanczos steps: " << lanczos_steps << "\n";
  std::cout << "Ground state energy: " << eigenpair.value << "\n";
  std::cout << "Residual norm: " << residual_norm << "\n";

  return 0;
}
