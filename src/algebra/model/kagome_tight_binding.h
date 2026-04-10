#pragma once

#include <cstddef>

#include "algebra/fermion/expression.h"
#include "algebra/fermion/model/model.h"
#include "utils/index.h"

/// Non-interacting tight-binding model on a 2D kagome lattice with periodic
/// boundaries.
///
/// The lattice has three sites per unit cell (A, B, C) on an Nx x Ny cluster.
/// With lattice vectors a1 = (1,0) and a2 = (1/2, sqrt(3)/2), sublattice
/// positions are A = (0,0), B = a1/2, C = a2/2.
///
/// Nearest-neighbor bonds (all hopping t):
///   Intra-cell:   A--B, A--C, B--C
///   Inter-cell:   B(ix,iy) -- A(ix+1, iy)
///                 C(ix,iy) -- A(ix, iy+1)
///                 C(ix,iy) -- B(ix-1, iy+1)
///
/// Flat-band energy: E_fb = +2t  (H uses -t convention)
///
/// Site indexing uses Index({sublattice, ix, iy}).
struct KagomeTightBindingModel : FermionModel {
  static constexpr size_t SUBLATTICE_A = 0;
  static constexpr size_t SUBLATTICE_B = 1;
  static constexpr size_t SUBLATTICE_C = 2;
  static constexpr auto spin = FermionOperator::Spin::Up;

  KagomeTightBindingModel(double t_val, size_t nx_val, size_t ny_val)
      : t(t_val),
        nx(nx_val),
        ny(ny_val),
        num_cells(nx_val * ny_val),
        num_sites(3 * nx_val * ny_val),
        index({3, nx_val, ny_val}) {}

  size_t site_A(size_t ix, size_t iy) const { return index({SUBLATTICE_A, ix, iy}); }
  size_t site_B(size_t ix, size_t iy) const { return index({SUBLATTICE_B, ix, iy}); }
  size_t site_C(size_t ix, size_t iy) const { return index({SUBLATTICE_C, ix, iy}); }

  FermionExpression hamiltonian() const override {
    FermionExpression result;
    const auto coeff = FermionExpression::complex_type(-t, 0.0);

    for (size_t ix = 0; ix < nx; ++ix) {
      for (size_t iy = 0; iy < ny; ++iy) {
        const size_t a = site_A(ix, iy);
        const size_t b = site_B(ix, iy);
        const size_t c = site_C(ix, iy);

        // Intra-cell: A--B, A--C, B--C
        result += coeff * hopping(a, b, spin);
        result += coeff * hopping(a, c, spin);
        result += coeff * hopping(b, c, spin);

        // Inter-cell: B(ix,iy) -- A(ix+1, iy)
        const size_t a_xnext = site_A((ix + 1) % nx, iy);
        result += coeff * hopping(b, a_xnext, spin);

        // Inter-cell: C(ix,iy) -- A(ix, iy+1)
        const size_t a_ynext = site_A(ix, (iy + 1) % ny);
        result += coeff * hopping(c, a_ynext, spin);

        // Inter-cell: C(ix,iy) -- B(ix-1, iy+1)
        const size_t b_diag = site_B((ix + nx - 1) % nx, (iy + 1) % ny);
        result += coeff * hopping(c, b_diag, spin);
      }
    }
    return result;
  }

  /// Flat-band energy. The Hamiltonian uses -t convention (H = -t c†c),
  /// so E_fb = +2t.
  double flat_band_energy() const { return 2.0 * t; }

  double t;
  size_t nx;
  size_t ny;
  size_t num_cells;
  size_t num_sites;
  Index index;
};
