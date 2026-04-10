#pragma once

#include <cstddef>

#include "algebra/fermion/expression.h"
#include "algebra/fermion/model/model.h"
#include "utils/index.h"

/// Non-interacting tight-binding model on a 1D sawtooth lattice with periodic
/// boundaries.
///
/// The lattice has two sites per unit cell:
///   - Base sites form a 1D chain with hopping t_base
///   - Apex sites connect to the two neighboring base sites with hopping t_apex
///
/// Flat-band energy: E_fb = +t_apex^2 / t_base  (H uses -t convention)
///
/// Site indexing uses Index({sublattice, cell}) where
///   sublattice 0 = base, sublattice 1 = apex.
struct SawtoothTightBindingModel : FermionModel {
  static constexpr size_t SUBLATTICE_BASE = 0;
  static constexpr size_t SUBLATTICE_APEX = 1;
  static constexpr auto spin = FermionOperator::Spin::Up;

  SawtoothTightBindingModel(double t_base_val, double t_apex_val, size_t num_cells_val)
      : t_base(t_base_val),
        t_apex(t_apex_val),
        num_cells(num_cells_val),
        num_sites(2 * num_cells_val),
        index({2, num_cells_val}) {}

  size_t site_base(size_t cell) const { return index({SUBLATTICE_BASE, cell}); }

  size_t site_apex(size_t cell) const { return index({SUBLATTICE_APEX, cell}); }

  FermionExpression hamiltonian() const override {
    FermionExpression result;
    const auto t_base_coeff = FermionExpression::complex_type(-t_base, 0.0);
    const auto t_apex_coeff = FermionExpression::complex_type(-t_apex, 0.0);

    for (size_t cell = 0; cell < num_cells; ++cell) {
      const size_t base_here = site_base(cell);
      const size_t base_next = site_base((cell + 1) % num_cells);
      const size_t apex_here = site_apex(cell);

      // Base chain: base(cell) -- base(cell+1)
      result += t_base_coeff * hopping(base_here, base_next, spin);

      // Apex legs: apex(cell) -- base(cell), apex(cell) -- base(cell+1)
      result += t_apex_coeff * hopping(apex_here, base_here, spin);
      result += t_apex_coeff * hopping(apex_here, base_next, spin);
    }
    return result;
  }

  /// Flat-band energy. The Hamiltonian uses -t convention (H = -t c†c),
  /// so E_fb = +t_apex^2 / t_base.
  double flat_band_energy() const { return (t_apex * t_apex) / t_base; }

  double t_base;
  double t_apex;
  size_t num_cells;
  size_t num_sites;
  Index index;
};
