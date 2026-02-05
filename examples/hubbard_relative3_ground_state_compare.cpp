#include <algorithm>
#include <armadillo>
#include <cstddef>
#include <exception>
#include <iostream>
#include <vector>

#include "algebra/fermion/basis.h"
#include "algebra/fermion/matrix_elements.h"
#include "algebra/model/hubbard_model.h"
#include "cxxopts.hpp"
#include "numerics/hubbard_relative3_operators.h"
#include "numerics/lanczos.h"
#include "utils/canonicalize_momentum.h"
#include "utils/index.h"
#include "utils/tolerances.h"

struct CliOptions {
  size_t lattice_size = 2;
  int64_t total_momentum = 0;
  double t = 1.0;
  double U = 4.0;
  size_t lanczos_steps = 50;
};

CliOptions parse_cli_options(int argc, char** argv) {
  CliOptions o;

  cxxopts::Options cli("hubbard_relative3_ground_state_compare",
                       "Compare 3-particle ground state energies (sparse vs Lanczos)");
  // clang-format off
  cli.add_options()
      ("L,lattice-size", "Lattice size per dimension",  cxxopts::value(o.lattice_size)->default_value("3"))
      ("P,total-momentum", "Total momentum value",      cxxopts::value(o.total_momentum)->default_value("0"))
      ("t,hopping", "Hopping amplitude",                cxxopts::value(o.t)->default_value("-1.0"))
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

struct ProjectedOp : LinearOperator<arma::cx_vec> {
  const HubbardRelative3& H;

  explicit ProjectedOp(const HubbardRelative3& H_val) : H(H_val) {}

  arma::cx_vec apply(const arma::cx_vec& v) const override {
    return H.project_antisymmetric(H.apply(H.project_antisymmetric(v)));
  }

  size_t dimension() const override { return H.dimension(); }
};

int main(int argc, char** argv) {
  const CliOptions opts = parse_cli_options(argc, argv);
  validate_options(opts);

  const size_t L = opts.lattice_size;
  const size_t size_x = L;
  const size_t size_y = L;
  const size_t size_z = L;
  const std::vector<size_t> size{size_x, size_y, size_z};
  const std::vector<int64_t> total_momentum{opts.total_momentum, opts.total_momentum,
                                            opts.total_momentum};

  const size_t particles = 3;
  const int spin_projection = 1;

  HubbardModel3D hubbard(opts.t, opts.U, size_x, size_y, size_z);
  Basis basis = Basis::with_fixed_particle_number_and_spin(size_x * size_y * size_z, particles,
                                                           spin_projection);
  const Expression hamiltonian_full = hubbard.hamiltonian();
  arma::sp_cx_mat H = compute_matrix_elements<arma::sp_cx_mat>(basis, hamiltonian_full);

  arma::cx_vec sparse_vals;
  arma::cx_mat sparse_vecs;
  if (!arma::eigs_gen(sparse_vals, sparse_vecs, H, 1, "sr")) {
    std::cerr << "Sparse diagonalization failed." << std::endl;
    return 1;
  }

  const double sparse_ground = std::real(sparse_vals(0));

  HubbardRelative3 hamiltonian_rel(size, total_momentum, opts.t, opts.U);

  ProjectedOp hamiltonian_rel_proj(hamiltonian_rel);
  const size_t dimension = hamiltonian_rel_proj.dimension();
  const size_t lanczos_steps = std::min(opts.lanczos_steps, dimension);

  const auto eigenpair = find_min_eigenpair(hamiltonian_rel_proj, lanczos_steps);
  const arma::cx_vec residual =
      hamiltonian_rel_proj.apply(eigenpair.vector) - eigenpair.value * eigenpair.vector;
  const double residual_norm = arma::norm(residual);

  std::cout << "3D Hubbard model ground state (N=3, Sz=1)\n";
  std::cout << "L=" << L << ", t=" << opts.t << ", U=" << opts.U << "\n";
  std::cout << "basis_size=" << basis.set.size() << "\n";
  std::cout << "sparse_lowest_eigenvalue=" << sparse_ground << "\n";
  std::cout << "\n";
  std::cout << "Relative-coordinate Hubbard (3 particles, antisymmetric sector)\n";
  std::cout << "K=" << opts.total_momentum << ", dimension=" << dimension
            << ", lanczos_steps=" << lanczos_steps << "\n";
  std::cout << "lanczos_lowest_eigenvalue=" << eigenpair.value << "\n";
  std::cout << "residual_norm=" << residual_norm << "\n";
  std::cout << "energy_difference=" << (eigenpair.value - sparse_ground) << "\n";

  return 0;
}
