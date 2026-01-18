#include <armadillo>
#include <cmath>
#include <iostream>
#include <vector>

#include "algebra/basis.h"
#include "algebra/commutator.h"
#include "algebra/matrix_elements.h"
#include "algebra/model/hubbard_model.h"
#include "algorithms/schriffer_wolff.h"

arma::vec unique_eigenvalues(const arma::vec& values) {
  constexpr auto kTolerance = 0.01;
  std::vector<double> unique_values;
  unique_values.reserve(values.n_elem);

  for (arma::uword i = 0; i < values.n_elem; ++i) {
    const double current = values(i);
    if (unique_values.empty()) {
      unique_values.push_back(current);
      continue;
    }

    const double last = unique_values.back();
    if (std::abs(current - last) > kTolerance) {
      unique_values.push_back(current);
    }
  }

  arma::vec result(unique_values.size());
  for (arma::uword i = 0; i < unique_values.size(); ++i) {
    result(i) = unique_values[i];
  }
  return result;
}

int main() {
  const size_t lattice_size = 10;
  const size_t sw_particles = 2;
  const size_t max_particles = 4;
  const double hopping = 1.0;
  const double interaction = -10.0;
  const size_t iterations = 1000;
  const float cutoff = static_cast<float>(0.1 * std::abs(hopping * hopping / interaction));

  HubbardModel hubbard(hopping, interaction, lattice_size);
  Basis sw_basis = Basis::with_fixed_particle_number(lattice_size, sw_particles);

  Expression kinetic = hubbard.kinetic();
  Expression interaction_term = hubbard.interaction();
  Expression hamiltonian = kinetic + interaction_term;

  Expression generator = schriffer_wolff(kinetic, interaction_term, sw_basis, iterations);
  Expression effective_hamiltonian = BCH(generator, hamiltonian, 1.0, iterations);

  for (size_t truncatation = 2; truncatation <= max_particles; ++truncatation) {
    Expression truncated_effective = effective_hamiltonian;
    truncated_effective.truncate_by_size(2 * truncatation).truncate_by_norm(cutoff);

    {
      std::ostringstream oss;
      oss << "effective_hamiltonian_L=" << lattice_size << "_T=" << truncatation << ".txt";
      std::ofstream os(oss.str());
      for ([[maybe_unused]] const auto& [ops, coeff] : truncated_effective.hashmap) {
        os << truncated_effective.to_string() << "\n";
      }
    }
    for (size_t particles = 2; particles <= max_particles; ++particles) {
      Basis diag_basis = Basis::with_fixed_particle_number(lattice_size, particles);
      arma::cx_mat H_exact = compute_matrix_elements<arma::cx_mat>(diag_basis, hamiltonian);
      arma::cx_mat H_effective =
          compute_matrix_elements<arma::cx_mat>(diag_basis, truncated_effective);

      arma::vec exact_vals;
      arma::cx_mat exact_vecs;
      if (!arma::eig_sym(exact_vals, exact_vecs, H_exact)) {
        std::cerr << "Diagonalization failed for exact Hamiltonian.\n";
        return 1;
      }

      arma::vec effective_vals;
      arma::cx_mat effective_vecs;
      if (!arma::eig_sym(effective_vals, effective_vecs, H_effective)) {
        std::cerr << "Diagonalization failed for truncated effective Hamiltonian.\n";
        return 1;
      }

      std::ostringstream oss;
      oss << "data_L=" << lattice_size << "_T=" << truncatation << "_P=" << particles << ".txt";
      std::ofstream os(oss.str());
      os << unique_eigenvalues(effective_vals) << "\n";
    }
  }
}
