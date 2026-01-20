#include <armadillo>
#include <cstddef>
#include <exception>
#include <iomanip>
#include <iostream>
#include <vector>

#include "algebra/basis.h"
#include "algebra/matrix_elements.h"
#include "algebra/model/hubbard_model_momentum.h"
#include "algebra/relative_basis_transform.h"
#include "cxxopts.hpp"
#include "utils/index.h"

// Demonstrates the relative position basis transformation for the 2-particle 3D Hubbard model.
//
// The transformation converts from momentum basis |p_up, K-p_up> to relative position basis |r>.
// Key properties:
// - The transformation matrix U is unitary (preserves eigenvalues)
// - The on-site interaction localizes to r=0 in the relative basis
// - The kinetic energy becomes diagonal in momentum space but spreads in position space
struct CliOptions {
  size_t size_x = 2;
  size_t size_y = 2;
  size_t size_z = 2;
  size_t kx = 0;
  size_t ky = 0;
  size_t kz = 0;
  size_t particles = 2;
  int spin_projection = 0;
  double t = 1.0;
  double U = 4.0;
};

CliOptions parse_cli_options(int argc, char** argv) {
  CliOptions o;

  cxxopts::Options cli("relative_basis_transform_example",
                       "Relative basis transform for the 3D Hubbard model");
  // clang-format off
  cli.add_options()
      ("x,size-x", "Lattice size in x dimension",       cxxopts::value(o.size_x)->default_value("2"))
      ("y,size-y", "Lattice size in y dimension",       cxxopts::value(o.size_y)->default_value("2"))
      ("z,size-z", "Lattice size in z dimension",       cxxopts::value(o.size_z)->default_value("2"))
      ("kx", "Total momentum Kx component",             cxxopts::value(o.kx)->default_value("0"))
      ("ky", "Total momentum Ky component",             cxxopts::value(o.ky)->default_value("0"))
      ("kz", "Total momentum Kz component",             cxxopts::value(o.kz)->default_value("0"))
      ("N,particles", "Number of particles",            cxxopts::value(o.particles)->default_value("2"))
      ("S,spin", "Spin projection (n_up - n_down)",     cxxopts::value(o.spin_projection)->default_value("0"))
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
  if (opts.kx >= opts.size_x || opts.ky >= opts.size_y || opts.kz >= opts.size_z) {
    std::cerr << "Momentum components must be less than corresponding lattice dimensions.\n";
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
}

void print_coords(std::ostream& os, const std::vector<size_t>& coords) {
  os << "(";
  for (size_t i = 0; i < coords.size(); ++i) {
    if (i > 0) {
      os << ",";
    }
    os << coords[i];
  }
  os << ")";
}

int main(int argc, char** argv) {
  const CliOptions opts = parse_cli_options(argc, argv);
  validate_options(opts);

  const std::vector<size_t> size{opts.size_x, opts.size_y, opts.size_z};
  const std::vector<size_t> total_momentum{opts.kx, opts.ky, opts.kz};
  const size_t sites = opts.size_x * opts.size_y * opts.size_z;

  std::cout << "=== Relative Basis Transform Example (3D) ===\n\n";
  std::cout << "Parameters: L=(" << opts.size_x << "x" << opts.size_y << "x" << opts.size_z
            << "), K=(" << opts.kx << "," << opts.ky << "," << opts.kz << "), t=" << opts.t
            << ", U=" << opts.U << "\n";
  std::cout << "Particles: N=" << opts.particles << ", Sz=" << opts.spin_projection << "\n\n";

  // Build momentum basis for 2 opts.particles (1 up, 1 down) with fixed total momentum
  Index index(size);
  Basis momentum_basis = Basis::with_fixed_particle_number_spin_momentum(
      sites, opts.particles, opts.spin_projection, index, total_momentum);

  std::cout << "Momentum basis states (|p_up, p_down>):\n";
  for (size_t i = 0; i < momentum_basis.set.size(); ++i) {
    const auto& state = momentum_basis.set[i];
    const auto up_coords = index(state[0].value());
    const auto down_coords = index(state[1].value());
    std::cout << "  " << i << ": |";
    print_coords(std::cout, up_coords);
    std::cout << ", ";
    print_coords(std::cout, down_coords);
    std::cout << ">\n";
  }
  std::cout << "\n";

  // Build Hubbard Hamiltonian in momentum basis
  HubbardModelMomentum hubbard(opts.t, opts.U, size);
  arma::cx_mat H_mom = compute_matrix_elements<arma::cx_mat>(momentum_basis, hubbard.hamiltonian());

  std::cout << "Hamiltonian in momentum basis:\n" << arma::real(H_mom) << "\n";

  // Build transformation matrix
  auto result = relative_position_transform_with_index(momentum_basis, index);
  const arma::cx_mat& transform = result.transform;

  std::cout << "Transformation matrix (real part):\n" << arma::real(transform) << "\n";
  std::cout << "Transformation matrix (imag part):\n" << arma::imag(transform) << "\n";

  // Transform Hamiltonian to relative position basis
  arma::cx_mat H_rel = transform.t() * H_mom * transform;
  H_rel.clean(1000.0 * arma::datum::eps);

  std::cout << "Hamiltonian in relative position basis:\n" << arma::real(H_rel) << "\n";

  // Show that interaction localizes to r=0
  arma::cx_mat H_int_mom =
      compute_matrix_elements<arma::cx_mat>(momentum_basis, hubbard.interaction());
  arma::cx_mat H_int_rel = transform.t() * H_int_mom * transform;
  H_int_rel.clean(1000.0 * arma::datum::eps);

  std::cout << "Interaction in momentum basis (real part):\n" << arma::real(H_int_mom) << "\n";
  std::cout << "Interaction in relative basis (real part):\n" << arma::real(H_int_rel) << "\n";
  std::cout << "Note: Interaction localizes to H_int(0,0) = " << std::real(H_int_rel(0, 0))
            << " (r=0)\n\n";

  // Compare eigenvalues
  arma::vec eig_mom;
  arma::eig_sym(eig_mom, arma::cx_mat(H_mom));

  arma::vec eig_rel;
  arma::eig_sym(eig_rel, arma::cx_mat(H_rel));

  std::cout << std::fixed << std::setprecision(6);
  std::cout << "Eigenvalues (momentum basis): " << eig_mom.t();
  std::cout << "Eigenvalues (relative basis): " << eig_rel.t();
  std::cout << "Difference norm: " << arma::norm(eig_mom - eig_rel) << "\n";

  return 0;
}
