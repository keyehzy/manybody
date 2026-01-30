/// SSH Topological Marker Example - Periodic Boundary Conditions
///
/// This example computes the 1D topological marker for the SSH model
/// using periodic boundary conditions (PBC). The marker correctly identifies
/// the topological phase (winding number = 1) vs trivial phase (winding number = 0).
///
/// Three methods are compared:
/// 1. Exact diagonalization
/// 2. Kernel Polynomial Method (KPM)

#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numbers>

#include "algebra/model/ssh_model.h"
#include "cxxopts.hpp"
#include "numerics/topological_marker.h"

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

struct CliOptions {
  double t1 = 0.5;
  double t2 = 1.5;
  size_t num_cells = 50;
  size_t kpm_order = 100;
};

CliOptions parse_cli_options(int argc, char** argv) {
  CliOptions o;

  cxxopts::Options cli("ssh_topological_marker_pbc",
                       "SSH topological marker with periodic boundary conditions");
  // clang-format off
  cli.add_options()
      ("t1", "Intracell hopping amplitude", cxxopts::value(o.t1)->default_value("0.5"))
      ("t2", "Intercell hopping amplitude", cxxopts::value(o.t2)->default_value("1.5"))
      ("L,num-cells", "Number of unit cells", cxxopts::value(o.num_cells)->default_value("50"))
      ("M,kpm-order", "KPM expansion order", cxxopts::value(o.kpm_order)->default_value("100"))
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

void test_ssh_pbc(double t1, double t2, size_t num_cells, size_t kpm_order) {
  std::cout << "=== SSH Model (PBC) ===\n";
  std::cout << "t1 = " << t1 << ", t2 = " << t2 << ", L = " << num_cells << "\n";
  std::cout << "Expected winding number: " << (t2 > t1 ? 1 : 0) << "\n\n";

  // Build model with PBC
  SSHModel model(t1, t2, num_cells);
  arma::mat H = model.single_particle_hamiltonian();
  arma::mat W = ssh::build_chiral_operator(num_cells);
  auto X = ssh::build_position_operator_exp_cells(num_cells);

  // 1. Exact diagonalization
  topological::Marker1D exact(H, W, X);
  auto local_exact = exact.local_marker_cells();

  // 2. KPM
  topological::Marker1D_KPM kpm(H, W, X, kpm_order);
  auto local_kpm = kpm.local_marker_cells();

  std::cout << "Local markers:\n";
  std::cout << "Site | Exact      | KPM        |\n";
  std::cout << "-----|------------|------------|\n";
  for (size_t i = 0; i < num_cells; ++i) {
    std::cout << std::setw(4) << i << " | " << std::setw(10) << local_exact[i] << " | "
              << std::setw(10) << local_kpm[i] << " |\n";
  }
  std::cout << "\n";
}

int main(int argc, char** argv) {
  CliOptions opts = parse_cli_options(argc, argv);

  std::cout << std::fixed << std::setprecision(6);

  std::cout << "========================================\n";
  std::cout << "SSH Topological Marker - PBC\n";
  std::cout << "========================================\n\n";

  test_ssh_pbc(opts.t1, opts.t2, opts.num_cells, opts.kpm_order);
  std::cout << "----------------------------------------\n\n";
  test_ssh_pbc(opts.t2, opts.t1, opts.num_cells, opts.kpm_order);

  return 0;
}
