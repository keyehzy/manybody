#include <algorithm>
#include <armadillo>
#include <cmath>
#include <complex>
#include <cstddef>
#include <exception>
#include <iostream>
#include <vector>

#include "algebra/basis.h"
#include "algebra/fourier_transform.h"
#include "algebra/matrix_elements.h"
#include "algebra/model/hubbard_model_momentum.h"
#include "cxxopts.hpp"

namespace {
constexpr double kRoundTripTolerance = 1e-2;

Expression::complex_type to_expression_complex(const std::complex<double>& value) {
  return Expression::complex_type(value.real(), value.imag());
}

Expression vector_to_expression(const arma::cx_vec& v, const Basis& basis) {
  Expression state;
  state.terms().reserve(basis.set.size());
  for (size_t i = 0; i < basis.set.size(); ++i) {
    const std::complex<double> coeff = v(i);
    if (coeff != std::complex<double>{}) {
      state.terms().emplace(basis.set[i], to_expression_complex(coeff));
    }
  }
  return state;
}

double max_expression_delta_norm(const Expression& lhs, const Expression& rhs) {
  double max_norm = 0.0;
  for (const auto& [ops, coeff] : lhs.terms()) {
    auto it = rhs.terms().find(ops);
    const auto rhs_coeff = (it == rhs.terms().end()) ? Expression::complex_type{} : it->second;
    const auto delta = coeff - rhs_coeff;
    max_norm = std::max(max_norm, std::norm(delta));
  }
  for (const auto& [ops, coeff] : rhs.terms()) {
    if (lhs.terms().find(ops) == lhs.terms().end()) {
      max_norm = std::max(max_norm, std::norm(coeff));
    }
  }
  return max_norm;
}
}  // namespace

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

  cxxopts::Options cli(
      "hubbard_3d_momentum_ground_state_real_space",
      "Compute ground state in momentum space and Fourier transform it to real space");
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

  if (basis.set.empty()) {
    std::cerr << "No basis states for the chosen particle number, spin, and momentum.\n";
    return 1;
  }

  const Expression hamiltonian = hubbard.hamiltonian();
  arma::sp_cx_mat H = compute_matrix_elements<arma::sp_cx_mat>(basis, hamiltonian);

  arma::cx_vec eigenvalues;
  arma::cx_mat eigenvectors;
  if (!arma::eigs_gen(eigenvalues, eigenvectors, H, 1, "sr")) {
    std::cerr << "Sparse diagonalization failed." << std::endl;
    return 1;
  }

  const arma::cx_vec ground_state = eigenvectors.col(0);
  const Expression momentum_state = normal_order(vector_to_expression(ground_state, basis));

  const Expression real_space_state = normal_order(transform_expression(
      fourier_transform_operator, momentum_state, index, FourierMode::Inverse));
  const Expression round_trip = normal_order(transform_expression(
      fourier_transform_operator, real_space_state, index, FourierMode::Direct));

  const double max_error_norm = max_expression_delta_norm(round_trip, momentum_state);
  const double tol_norm = kRoundTripTolerance * kRoundTripTolerance;
  if (max_error_norm > tol_norm) {
    std::cerr << "Round-trip Fourier transform check failed. "
              << "max_error=" << std::sqrt(max_error_norm) << ", tolerance=" << kRoundTripTolerance
              << "\n";
    return 1;
  }

  std::cout << "3D Hubbard model ground state (momentum -> real space)\n";
  std::cout << "L=(" << opts.size_x << "x" << opts.size_y << "x" << opts.size_z << ")"
            << ", N=" << opts.particles << ", Sz=" << opts.spin_projection << ", K=(" << opts.kx
            << "," << opts.ky << "," << opts.kz << ")\n";
  std::cout << "t=" << opts.t << ", U=" << opts.U << "\n";
  std::cout << "Basis size: " << basis.set.size() << "\n";
  std::cout << "Lowest eigenvalue (sparse): " << eigenvalues(0) << "\n";
  std::cout << "Round-trip Fourier transform check: ok (max_error=" << std::sqrt(max_error_norm)
            << ")\n";
  std::cout << "Real-space ground state expression:\n";
  std::cout << real_space_state.to_string() << "\n";

  return 0;
}
