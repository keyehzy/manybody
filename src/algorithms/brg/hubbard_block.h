#pragma once

#include <complex>
#include <cstddef>
#include <utility>
#include <vector>

#include "algebra/expression.h"

namespace brg {

/// Geometry specification for a Hubbard block.
struct BlockGeometry {
  size_t num_sites;
  std::vector<std::pair<size_t, size_t>> bonds;
  std::vector<size_t> border_sites;  // sites on boundary for inter-block coupling
  double nu;                         // number of inter-block couplings per direction/face
};

/// 2D 2x2 block geometry (open boundary conditions).
///
/// Sites:  0:(0,0)  1:(0,1)
///         2:(1,0)  3:(1,1)
/// Bonds:  (0-1), (0-2), (1-3), (2-3)
/// Border: {1, 3} (right edge for +x coupling)
/// nu = 2 (two couplings per direction)
inline BlockGeometry block_2d_2x2() {
  return BlockGeometry{
      .num_sites = 4,
      .bonds = {{0, 1}, {0, 2}, {1, 3}, {2, 3}},
      .border_sites = {1, 3},
      .nu = 2.0,
  };
}

/// 3D 2x2x2 block geometry (open boundary conditions).
///
/// Sites (index = 4*z + 2*y + x):
///   z=0 layer:        z=1 layer:
///     2---3              6---7
///     |   |              |   |
///     0---1              4---5
///
/// Bonds (12 total):
///   x: (0,1), (2,3), (4,5), (6,7)
///   y: (0,2), (1,3), (4,6), (5,7)
///   z: (0,4), (1,5), (2,6), (3,7)
///
/// Border: {1, 3, 5, 7} (right +x face)
/// nu = 4 (four couplings per face)
inline BlockGeometry block_3d_2x2x2() {
  return BlockGeometry{
      .num_sites = 8,
      .bonds =
          {
              // x-direction
              {0, 1},
              {2, 3},
              {4, 5},
              {6, 7},
              // y-direction
              {0, 2},
              {1, 3},
              {4, 6},
              {5, 7},
              // z-direction
              {0, 4},
              {1, 5},
              {2, 6},
              {3, 7},
          },
      .border_sites = {1, 3, 5, 7},
      .nu = 4.0,
  };
}

/// Build Hubbard block Hamiltonian for a given geometry.
///
/// H = -t Σ_{<i,j>,σ} (c†_iσ c_jσ + h.c.)
///   + U Σ_i n_{i↑} n_{i↓}
///   - μ Σ_{i,σ} n_{iσ}
inline Expression build_hubbard_block_hamiltonian(const BlockGeometry& geometry, double t, double U,
                                                  double mu) {
  Expression H;

  // Hopping: -t * sum_{<i,j>,sigma} (c^dag_i c_j + h.c.)
  for (auto [i, j] : geometry.bonds) {
    for (auto sigma : {Operator::Spin::Up, Operator::Spin::Down}) {
      H += hopping({-t, 0.0}, i, j, sigma);
    }
  }

  // On-site interaction: U * sum_i n_{i,up} n_{i,down}
  for (size_t i = 0; i < geometry.num_sites; ++i) {
    H += Expression(density_density(Operator::Spin::Up, i, Operator::Spin::Down, i)) *
         std::complex<double>(U, 0.0);
  }

  // Chemical potential: -mu * sum_i (n_{i,up} + n_{i,down})
  for (size_t i = 0; i < geometry.num_sites; ++i) {
    for (auto sigma : {Operator::Spin::Up, Operator::Spin::Down}) {
      H += Expression(density(sigma, i)) * std::complex<double>(-mu, 0.0);
    }
  }

  return H;
}

/// Convenience: Build 2D 2x2 Hubbard block Hamiltonian.
inline Expression build_2d_block_hamiltonian(double t, double U, double mu) {
  return build_hubbard_block_hamiltonian(block_2d_2x2(), t, U, mu);
}

/// Convenience: Build 3D 2x2x2 Hubbard block Hamiltonian.
inline Expression build_3d_block_hamiltonian(double t, double U, double mu) {
  return build_hubbard_block_hamiltonian(block_3d_2x2x2(), t, U, mu);
}

/// Pair creation operator: Delta^dag_i = c^dag_{i,up} c^dag_{i,down}
/// Creates a singlet Cooper pair at site i.
inline Expression pair_creation(size_t site) {
  FermionMonomial t({Operator::creation(Operator::Spin::Up, site),
                     Operator::creation(Operator::Spin::Down, site)});
  return Expression(t);
}

/// Pair annihilation operator: Delta_i = c_{i,down} c_{i,up}
/// Annihilates a singlet Cooper pair at site i.
inline Expression pair_annihilation(size_t site) {
  FermionMonomial t({Operator::annihilation(Operator::Spin::Down, site),
                     Operator::annihilation(Operator::Spin::Up, site)});
  return Expression(t);
}

}  // namespace brg
