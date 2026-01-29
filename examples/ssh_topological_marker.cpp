/// SSH Model Topological Marker: KPM vs Lanczos-Ritz Comparison
///
/// This example compares:
/// 1. Kernel Polynomial Method (KPM) - fixed polynomial order
/// 2. Lanczos-Ritz method - adaptive Krylov subspace
///
/// For the SSH model (class AIII), we compute the 1D topological marker
/// following the universal marker construction (Paper 2). The spatially
/// averaged marker should reproduce the winding number in the thermodynamic
/// limit.
///
/// Hypothesis: The Krylov dimension required for convergence scales with
/// the bulk gap. Deep in the topological/trivial phase (large gap), we need
/// fewer iterations. Near the critical point (gap → 0), we need more.

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numbers>
#include <vector>

#include "algebra/model/ssh_model.h"
#include "numerics/topological_marker.h"

void print_separator() { std::cout << std::string(70, '-') << "\n"; }

/// Compare exact, KPM, and Lanczos marker averages for a single parameter point
void compare_methods(double t1, double t2, size_t num_cells, size_t kpm_order,
                     size_t lanczos_steps) {
  SSHModel model(t1, t2, num_cells);

  std::cout << "SSH Model: t1 = " << t1 << ", t2 = " << t2 << ", L = " << num_cells << "\n";
  std::cout << "Bulk gap = " << model.bulk_gap() << ", ξ = " << model.correlation_length() << "\n";
  std::cout << "Expected winding number = " << model.winding_number() << "\n\n";

  arma::mat H = model.single_particle_hamiltonian();
  arma::mat W = ssh::build_chiral_operator(num_cells);
  arma::cx_mat X = ssh::build_position_operator_exp_cells(num_cells);

  // 1. Exact calculation using Marker1D
  topological::Marker1D exact(H, W, X);
  double exact_marker = exact.average_marker() / std::numbers::pi;
  std::cout << "Exact marker (avg / π): " << std::fixed << std::setprecision(4) << exact_marker
            << "\n";

  // 2. KPM calculation
  topological::Marker1D_KPM kpm_marker(H, W, X, kpm_order);
  double kpm_value = kpm_marker.average_marker() / std::numbers::pi;
  double kpm_error = std::abs(kpm_value - exact_marker);
  std::cout << "KPM marker (avg / π, M=" << kpm_order << "): " << kpm_value
            << " (error: " << kpm_error << ")\n";

  // 3. Lanczos calculation
  topological::Marker1D_Lanczos lanczos_marker(H, W, X);
  auto [lanczos_value_raw, krylov_dim] = lanczos_marker.average_marker(lanczos_steps);
  double lanczos_value = lanczos_value_raw / std::numbers::pi;
  double lanczos_error = std::abs(lanczos_value - exact_marker);
  std::cout << "Lanczos marker (avg / π, m=" << krylov_dim << "): " << lanczos_value
            << " (error: " << lanczos_error << ")\n";
}

/// Study convergence of KPM as a function of expansion order
void study_kpm_convergence(double t1, double t2, size_t num_cells, const std::string& output_file) {
  SSHModel model(t1, t2, num_cells);
  arma::mat H = model.single_particle_hamiltonian();
  arma::mat W = ssh::build_chiral_operator(num_cells);
  arma::cx_mat X = ssh::build_position_operator_exp_cells(num_cells);

  // Exact reference
  topological::Marker1D exact(H, W, X);
  double exact_marker = exact.average_marker() / std::numbers::pi;

  std::ofstream file(output_file);
  file << "# KPM convergence for SSH model marker (avg/pi)\n";
  file << "# t1 = " << t1 << ", t2 = " << t2 << ", L = " << num_cells << "\n";
  file << "# gap = " << model.bulk_gap() << ", exact_marker/pi = " << exact_marker << "\n";
  file << "# M, marker, error\n";

  std::cout << "KPM convergence study (gap = " << model.bulk_gap() << "):\n";
  std::cout << std::setw(10) << "M" << std::setw(15) << "marker/pi" << std::setw(15) << "error\n";

  for (size_t M = 10; M <= 200; M += 10) {
    topological::Marker1D_KPM kpm_marker(H, W, X, M);
    double marker = kpm_marker.average_marker() / std::numbers::pi;
    double error = std::abs(marker - exact_marker);

    file << M << " " << marker << " " << error << "\n";
    std::cout << std::setw(10) << M << std::setw(15) << std::fixed << std::setprecision(6) << marker
              << std::setw(15) << std::scientific << error << "\n";
  }

  file.close();
  std::cout << "Results written to " << output_file << "\n";
}

/// Study convergence of Lanczos as a function of Krylov dimension
void study_lanczos_convergence(double t1, double t2, size_t num_cells,
                               const std::string& output_file) {
  SSHModel model(t1, t2, num_cells);
  arma::mat H = model.single_particle_hamiltonian();
  arma::mat W = ssh::build_chiral_operator(num_cells);
  arma::cx_mat X = ssh::build_position_operator_exp_cells(num_cells);

  // Exact reference
  topological::Marker1D exact(H, W, X);
  double exact_marker = exact.average_marker() / std::numbers::pi;

  std::ofstream file(output_file);
  file << "# Lanczos convergence for SSH model marker (avg/pi)\n";
  file << "# t1 = " << t1 << ", t2 = " << t2 << ", L = " << num_cells << "\n";
  file << "# gap = " << model.bulk_gap() << ", exact_marker/pi = " << exact_marker << "\n";
  file << "# m, marker, error\n";

  std::cout << "Lanczos convergence study (gap = " << model.bulk_gap() << "):\n";
  std::cout << std::setw(10) << "m" << std::setw(15) << "marker/pi" << std::setw(15) << "error\n";

  topological::Marker1D_Lanczos lanczos_marker(H, W, X);

  for (size_t m = 10; m <= std::min<size_t>(200, model.num_sites); m += 10) {
    auto [marker_raw, actual_m] = lanczos_marker.average_marker(m);
    double marker = marker_raw / std::numbers::pi;
    double error = std::abs(marker - exact_marker);

    file << actual_m << " " << marker << " " << error << "\n";
    std::cout << std::setw(10) << actual_m << std::setw(15) << std::fixed << std::setprecision(6)
              << marker << std::setw(15) << std::scientific << error << "\n";
  }

  file.close();
  std::cout << "Results written to " << output_file << "\n";
}

/// Main hypothesis test: Krylov complexity vs bulk gap
void study_krylov_complexity(size_t num_cells, const std::string& output_file) {
  std::ofstream file(output_file);
  file << "# Krylov complexity vs bulk gap for SSH model\n";
  file << "# L = " << num_cells << "\n";
  file << "# delta (t1-t2), gap, krylov_dim_converged, exact_marker/pi, converged_marker/pi\n";

  std::cout << "\nKrylov Complexity Analysis:\n";
  std::cout << std::setw(12) << "delta" << std::setw(12) << "gap" << std::setw(15) << "krylov_dim"
            << std::setw(15) << "exact/pi" << std::setw(15) << "computed/pi" << std::setw(12)
            << "converged\n";
  print_separator();

  // Scan from deep topological through critical to deep trivial
  std::vector<double> t2_values;

  // Far from critical (topological)
  for (double t2 = 2.0; t2 > 1.2; t2 -= 0.2) t2_values.push_back(t2);
  // Near critical
  for (double t2 = 1.2; t2 > 0.8; t2 -= 0.05) t2_values.push_back(t2);
  // Far from critical (trivial)
  for (double t2 = 0.8; t2 >= 0.2; t2 -= 0.2) t2_values.push_back(t2);

  const double t1 = 1.0;

  for (double t2 : t2_values) {
    SSHModel model(t1, t2, num_cells);
    arma::mat H = model.single_particle_hamiltonian();
    arma::mat W = ssh::build_chiral_operator(num_cells);
    arma::cx_mat X = ssh::build_position_operator_exp_cells(num_cells);

    // Exact marker
    topological::Marker1D exact(H, W, X);
    double exact_marker = exact.average_marker() / std::numbers::pi;

    // Lanczos with convergence tracking
    topological::Marker1D_Lanczos lanczos_marker(H, W, X);
    auto result = lanczos_marker.compute_with_convergence(1e-4 * std::numbers::pi, model.num_sites);
    double converged_marker = result.marker / std::numbers::pi;

    double delta = t1 - t2;
    file << delta << " " << model.bulk_gap() << " " << result.krylov_dim << " " << exact_marker
         << " " << converged_marker << "\n";

    std::cout << std::setw(12) << std::fixed << std::setprecision(2) << delta << std::setw(12)
              << model.bulk_gap() << std::setw(15) << result.krylov_dim << std::setw(15)
              << std::setprecision(4) << exact_marker << std::setw(15) << converged_marker
              << std::setw(12) << (result.converged ? "yes" : "no") << "\n";
  }

  file.close();
  std::cout << "\nResults written to " << output_file << "\n";
}

/// Study the spatial profile of the local marker
void study_local_marker_profile(double t1, double t2, size_t num_cells,
                                const std::string& output_file) {
  SSHModel model(t1, t2, num_cells);
  arma::mat H = model.single_particle_hamiltonian();
  arma::mat W = ssh::build_chiral_operator(num_cells);
  arma::cx_mat X = ssh::build_position_operator_exp_cells(num_cells);

  topological::Marker1D exact(H, W, X);
  auto local = exact.local_marker();

  std::ofstream file(output_file);
  file << "# Local topological marker profile for SSH model\n";
  file << "# t1 = " << t1 << ", t2 = " << t2 << ", L = " << num_cells << "\n";
  file << "# Total marker (avg/pi) = " << exact.average_marker() / std::numbers::pi << "\n";
  file << "# unit_cell, marker\n";

  double total = 0.0;
  for (size_t r = 0; r < num_cells; ++r) {
    const size_t A = 2 * r;
    const size_t B = 2 * r + 1;
    double cell_marker = local[A] + local[B];
    file << r << " " << cell_marker << "\n";
    total += cell_marker;
  }

  file.close();
  std::cout << "Local marker profile written to " << output_file << "\n";
  std::cout << "  Total (sum of local markers): " << total << "\n";
  std::cout << "  Marker average (/π): " << exact.average_marker() / std::numbers::pi << "\n";
}

/// Compare efficiency: operations count for KPM vs Lanczos to reach given accuracy
void compare_efficiency(size_t num_cells, double target_accuracy, const std::string& output_file) {
  std::ofstream file(output_file);
  file << "# Efficiency comparison: KPM vs Lanczos for marker (avg/pi)\n";
  file << "# L = " << num_cells << ", target_accuracy = " << target_accuracy << "\n";
  file << "# gap, kpm_order, lanczos_dim, kpm_ops, lanczos_ops\n";

  std::cout << "\nEfficiency Comparison (target accuracy = " << target_accuracy << "):\n";
  std::cout << std::setw(10) << "gap" << std::setw(12) << "KPM_M" << std::setw(12) << "Lanczos_m"
            << std::setw(15) << "KPM_ops" << std::setw(15) << "Lanczos_ops\n";
  print_separator();

  const double t1 = 1.0;
  std::vector<double> gaps = {2.0, 1.5, 1.0, 0.5, 0.2, 0.1};

  for (double gap : gaps) {
    double t2 = t1 + gap / 2.0;  // gap = 2|t1 - t2|

    SSHModel model(t1, t2, num_cells);
    arma::mat H = model.single_particle_hamiltonian();
    arma::mat W = ssh::build_chiral_operator(num_cells);
    arma::cx_mat X = ssh::build_position_operator_exp_cells(num_cells);

    // Exact reference
    topological::Marker1D exact(H, W, X);
    double exact_marker = exact.average_marker() / std::numbers::pi;

    // Find KPM order needed for target accuracy
    size_t kpm_order = 0;
    for (size_t M = 10; M <= 500; M += 5) {
      topological::Marker1D_KPM kpm_marker(H, W, X, M);
      double marker = kpm_marker.average_marker() / std::numbers::pi;
      double error = std::abs(marker - exact_marker);
      if (error < target_accuracy) {
        kpm_order = M;
        break;
      }
    }

    // Find Lanczos dim needed for target accuracy
    topological::Marker1D_Lanczos lanczos_marker(H, W, X);
    auto result = lanczos_marker.compute_with_convergence(target_accuracy * std::numbers::pi,
                                                          model.num_sites);
    size_t lanczos_dim = result.krylov_dim;

    // Operations count (approximate):
    // KPM: M matrix-vector products
    // Lanczos: m matrix-vector products + O(m²) for tridiagonal eigenproblem
    size_t kpm_ops = kpm_order * model.num_sites;
    size_t lanczos_ops = lanczos_dim * model.num_sites + lanczos_dim * lanczos_dim;

    file << gap << " " << kpm_order << " " << lanczos_dim << " " << kpm_ops << " " << lanczos_ops
         << "\n";

    std::cout << std::setw(10) << std::fixed << std::setprecision(2) << gap << std::setw(12)
              << kpm_order << std::setw(12) << lanczos_dim << std::setw(15) << kpm_ops
              << std::setw(15) << lanczos_ops << "\n";
  }

  file.close();
  std::cout << "\nResults written to " << output_file << "\n";
}

/// Scan through phase diagram
void phase_diagram_scan(size_t num_cells, const std::string& output_file) {
  std::ofstream file(output_file);
  file << "# Phase diagram scan for SSH model\n";
  file << "# L = " << num_cells << "\n";
  file << "# t2/t1, exact_marker/pi, kpm_marker/pi, lanczos_marker/pi\n";

  std::cout << "\nPhase Diagram Scan:\n";
  std::cout << std::setw(10) << "t2/t1" << std::setw(15) << "exact/pi" << std::setw(15) << "KPM/pi"
            << std::setw(15) << "Lanczos/pi\n";
  print_separator();

  const double t1 = 1.0;

  for (double ratio = 0.2; ratio <= 3.0; ratio += 0.2) {
    double t2 = t1 * ratio;
    SSHModel model(t1, t2, num_cells);
    arma::mat H = model.single_particle_hamiltonian();
    arma::mat W = ssh::build_chiral_operator(num_cells);
    arma::cx_mat X = ssh::build_position_operator_exp_cells(num_cells);

    topological::Marker1D exact(H, W, X);
    topological::Marker1D_KPM kpm(H, W, X, 100);
    topological::Marker1D_Lanczos lanczos(H, W, X);

    double exact_v = exact.average_marker() / std::numbers::pi;
    double kpm_v = kpm.average_marker() / std::numbers::pi;
    auto [lanczos_v_raw, m] = lanczos.average_marker(100);
    double lanczos_v = lanczos_v_raw / std::numbers::pi;

    file << ratio << " " << exact_v << " " << kpm_v << " " << lanczos_v << "\n";

    std::cout << std::setw(10) << std::fixed << std::setprecision(2) << ratio << std::setw(15)
              << std::setprecision(4) << exact_v << std::setw(15) << kpm_v << std::setw(15)
              << lanczos_v << "\n";
  }

  file.close();
  std::cout << "\nResults written to " << output_file << "\n";
}

int main() {
  std::cout << "===== SSH Topological Marker: KPM vs Lanczos-Ritz =====\n\n";

  const size_t num_cells = 50;

  // Create results directory
  std::string results_dir = "results/ssh_topological_marker";
  std::system(("mkdir -p " + results_dir).c_str());

  print_separator();
  std::cout << "1. Basic comparison at different parameter points\n";
  print_separator();

  std::cout << "\n[TOPOLOGICAL PHASE: t1 < t2]\n";
  compare_methods(0.5, 1.5, num_cells, 100, 100);

  std::cout << "\n[NEAR CRITICAL: t1 ≈ t2]\n";
  compare_methods(0.9, 1.1, num_cells, 100, 100);

  std::cout << "\n[TRIVIAL PHASE: t1 > t2]\n";
  compare_methods(1.5, 0.5, num_cells, 100, 100);

  print_separator();
  std::cout << "\n2. Phase Diagram Scan\n";
  print_separator();

  phase_diagram_scan(num_cells, results_dir + "/phase_diagram.dat");

  print_separator();
  std::cout << "\n3. KPM Convergence Studies\n";
  print_separator();

  study_kpm_convergence(0.5, 1.5, num_cells, results_dir + "/kpm_convergence_topological.dat");
  std::cout << "\n";
  study_kpm_convergence(0.9, 1.1, num_cells, results_dir + "/kpm_convergence_critical.dat");

  print_separator();
  std::cout << "\n4. Lanczos Convergence Studies\n";
  print_separator();

  study_lanczos_convergence(0.5, 1.5, num_cells,
                            results_dir + "/lanczos_convergence_topological.dat");
  std::cout << "\n";
  study_lanczos_convergence(0.9, 1.1, num_cells, results_dir + "/lanczos_convergence_critical.dat");

  print_separator();
  std::cout << "\n5. Krylov Complexity vs Bulk Gap (Main Hypothesis Test)\n";
  print_separator();

  study_krylov_complexity(num_cells, results_dir + "/krylov_complexity.dat");

  print_separator();
  std::cout << "\n6. Local Marker Spatial Profile\n";
  print_separator();

  study_local_marker_profile(0.5, 1.5, num_cells, results_dir + "/local_marker_topological.dat");
  study_local_marker_profile(1.5, 0.5, num_cells, results_dir + "/local_marker_trivial.dat");

  print_separator();
  std::cout << "\n7. Efficiency Comparison\n";
  print_separator();

  compare_efficiency(num_cells, 1e-3, results_dir + "/efficiency_comparison.dat");

  std::cout << "\n===== Analysis Complete =====\n";
  std::cout << "Results saved in: " << results_dir << "/\n";

  return 0;
}
