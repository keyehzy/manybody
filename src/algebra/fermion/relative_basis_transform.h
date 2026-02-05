#pragma once

#include <armadillo>
#include <cassert>
#include <cmath>
#include <complex>
#include <cstddef>
#include <numbers>
#include <vector>

#include "algebra/fermion/basis.h"
#include "algebra/fermion/operator.h"
#include "utils/index.h"

// Constructs the unitary transformation matrix from momentum basis to relative position basis.
//
// For N particles with fixed total momentum K, there are N-1 relative coordinates.
// The transformation is a multi-dimensional Fourier transform:
//
//   |r_1, ..., r_{N-1}; K⟩ = (1/L^{(N-1)/2}) Σ exp(i Σ_j p_j · r_j) |p_1, ..., p_{N-1}; K⟩
//
// where:
//   - p_j is the momentum of the j-th non-reference particle
//   - r_j is the relative position of the j-th non-reference particle
//   - The reference particle's momentum p_0 = K - Σ p_j is fixed by momentum conservation
//
// The momentum basis is obtained from Basis::with_fixed_particle_number_spin_momentum().
// The relative position basis has dimension L^{N-1} (same as momentum basis).
//
// Parameters:
//   momentum_basis: Basis with fixed particle number, spin projection, and total momentum
//   index: Index object mapping orbital indices to/from momentum coordinates
//
// Returns:
//   Complex unitary matrix U where U[rel_pos_idx, mom_basis_idx] is the Fourier coefficient.
//   To transform an operator H from momentum to relative position: H_rel = U† * H_mom * U
inline arma::cx_mat relative_position_transform(const Basis& momentum_basis, const Index& index) {
  const size_t basis_size = momentum_basis.set.size();
  if (basis_size == 0) {
    return arma::cx_mat();
  }

  const size_t num_particles = momentum_basis.particles;
  if (num_particles < 2) {
    throw std::invalid_argument(
        "relative_position_transform requires at least 2 particles for relative coordinates");
  }

  const size_t num_relative = num_particles - 1;
  const size_t num_orbitals = index.size();
  const auto& dims = index.dimensions();

  // The relative position basis has dimension num_orbitals^{num_relative}
  // This should match the momentum basis size
  size_t expected_size = 1;
  for (size_t i = 0; i < num_relative; ++i) {
    expected_size *= num_orbitals;
  }

  // Build Index for relative coordinates (stacked dimensions)
  std::vector<size_t> relative_dims;
  relative_dims.reserve(dims.size() * num_relative);
  for (size_t i = 0; i < num_relative; ++i) {
    relative_dims.insert(relative_dims.end(), dims.begin(), dims.end());
  }
  Index relative_index(relative_dims);

  // Normalization factor: 1/L^{(N-1)/2} where L = num_orbitals
  const double normalization = 1.0 / std::sqrt(static_cast<double>(expected_size));

  // Allocate transformation matrix
  arma::cx_mat U(expected_size, basis_size, arma::fill::zeros);

  // For each momentum basis state, compute Fourier coefficients to all relative positions
  for (size_t mom_idx = 0; mom_idx < basis_size; ++mom_idx) {
    const auto& state = momentum_basis.set[mom_idx];

    // Extract momenta for Fourier transform
    // For the 2-particle case with 1 up-spin and 1 down-spin:
    //   B^+_{K,p} = c^+_{p,up} c^+_{K-p,down}
    //   B^+_{K,r} = (1/sqrt(N)) sum_p exp(i p r) B^+_{K,p}
    // The Fourier variable is the UP-spin momentum p, not the DOWN-spin momentum.
    //
    // More generally, we use the FIRST particle's momentum as the Fourier variable.
    // state contains creation operators in canonical order (up-spins first, then down-spins)
    std::vector<size_t> fourier_momenta;
    fourier_momenta.reserve(num_relative);

    // Use the first particle's momentum as the Fourier variable
    // For 2 particles: use state[0] (the up-spin momentum)
    // For N particles: this generalizes to using the first N-1 particles
    for (size_t p = 0; p < num_relative; ++p) {
      fourier_momenta.push_back(state[p].value());
    }

    assert(fourier_momenta.size() == num_relative);

    // For each relative position configuration
    for (size_t rel_idx = 0; rel_idx < expected_size; ++rel_idx) {
      // Get relative position coordinates
      const auto rel_coords = relative_index(rel_idx);

      // Compute phase: Σ_j p_j · r_j
      double phase = 0.0;
      size_t coord_offset = 0;

      for (size_t j = 0; j < num_relative; ++j) {
        // Get momentum coordinates for particle j
        const auto mom_coords = index(fourier_momenta[j]);

        // Compute p_j · r_j contribution
        for (size_t d = 0; d < dims.size(); ++d) {
          const double p_d = static_cast<double>(mom_coords[d]);
          const double r_d = static_cast<double>(rel_coords[coord_offset + d]);
          const double L_d = static_cast<double>(dims[d]);
          phase += 2.0 * std::numbers::pi_v<double> * p_d * r_d / L_d;
        }
        coord_offset += dims.size();
      }

      // U[rel_idx, mom_idx] = normalization * exp(i * phase)
      U(rel_idx, mom_idx) = normalization * std::exp(std::complex<double>(0.0, phase));
    }
  }

  return U;
}

// Overload that builds the relative Index internally and returns it along with the transform
struct RelativeTransformResult {
  arma::cx_mat transform;      // Unitary transformation matrix
  Index relative_index;        // Index for relative position coordinates
  size_t num_relative_coords;  // Number of relative coordinates (N-1 for N particles)
};

inline RelativeTransformResult relative_position_transform_with_index(const Basis& momentum_basis,
                                                                      const Index& index) {
  const size_t num_particles = momentum_basis.particles;
  if (num_particles < 2) {
    throw std::invalid_argument(
        "relative_position_transform requires at least 2 particles for relative coordinates");
  }

  const size_t num_relative = num_particles - 1;
  const auto& dims = index.dimensions();

  // Build Index for relative coordinates (stacked dimensions)
  std::vector<size_t> relative_dims;
  relative_dims.reserve(dims.size() * num_relative);
  for (size_t i = 0; i < num_relative; ++i) {
    relative_dims.insert(relative_dims.end(), dims.begin(), dims.end());
  }

  return RelativeTransformResult{
      .transform = relative_position_transform(momentum_basis, index),
      .relative_index = Index(relative_dims),
      .num_relative_coords = num_relative,
  };
}
