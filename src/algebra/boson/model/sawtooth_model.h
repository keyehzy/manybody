#pragma once

#include <cstddef>

#include "algebra/boson/expression.h"
#include "algebra/model.h"
#include "utils/index.h"

/// Single-component Bose-Hubbard model on a 1D sawtooth lattice with periodic boundaries.
///
/// The lattice has two sites per unit cell:
///   - Base sites form a 1D chain with hopping t_base
///   - Apex sites connect to the two neighboring base sites with hopping t_tooth
///
/// The on-site interaction is the standard Bose-Hubbard term
///   (U / 2) * sum_i n_i (n_i - 1)
/// written in normal-ordered form as (U / 2) * b_i^+ b_i^+ b_i b_i.
///
/// This is a single-component (spinless) model. The spin degree of freedom is
/// repurposed as a species label, with all particles using Spin::Up.
struct SawtoothHubbardModel : BasicModel<BosonExpression> {
  static constexpr size_t SUBLATTICE_BASE = 0;
  static constexpr size_t SUBLATTICE_APEX = 1;
  static constexpr BosonOperator::Spin species = BosonOperator::Spin::Up;

  SawtoothHubbardModel(double t_base_val, double t_tooth_val, double u_val, size_t num_cells_val)
      : t_base(t_base_val),
        t_tooth(t_tooth_val),
        u(u_val),
        num_cells(num_cells_val),
        num_sites(2 * num_cells_val),
        index({2, num_cells_val}) {}

  size_t site(size_t sublattice, size_t cell) const { return index({sublattice, cell}); }

  size_t site(size_t sublattice, size_t cell, int cell_offset) const {
    return index({sublattice, cell}, {0, cell_offset});
  }

  size_t site_base(size_t cell, int cell_offset = 0) const {
    return site(SUBLATTICE_BASE, cell, cell_offset);
  }

  size_t site_apex(size_t cell, int cell_offset = 0) const {
    return site(SUBLATTICE_APEX, cell, cell_offset);
  }

  BosonExpression base_hopping() const {
    BosonExpression result;
    const auto coeff = BosonExpression::complex_type(-t_base, 0.0);
    for (size_t cell = 0; cell < num_cells; ++cell) {
      result += hopping(coeff, site_base(cell), site_base(cell, 1), species);
    }
    return result;
  }

  BosonExpression tooth_hopping() const {
    BosonExpression result;
    const auto coeff = BosonExpression::complex_type(-t_tooth, 0.0);
    for (size_t cell = 0; cell < num_cells; ++cell) {
      const size_t apex = site_apex(cell);
      result += hopping(coeff, site_base(cell), apex, species);
      result += hopping(coeff, site_base(cell, 1), apex, species);
    }
    return result;
  }

  BosonExpression kinetic() const {
    BosonExpression result = base_hopping();
    result += tooth_hopping();
    return result;
  }

  BosonExpression interaction() const {
    BosonExpression result;
    const auto coeff = BosonExpression::complex_type(0.5 * u, 0.0);
    for (size_t cell = 0; cell < num_cells; ++cell) {
      result += coeff * onsite_interaction(site_base(cell));
      result += coeff * onsite_interaction(site_apex(cell));
    }
    return result;
  }

  BosonExpression hamiltonian() const override {
    BosonExpression result = kinetic();
    result += interaction();
    return result;
  }

  /// On-site Hubbard interaction n_i(n_i - 1) = n_i^2 - n_i.
  /// density_density gives n_i^2, so we subtract n_i.
  static BosonExpression onsite_interaction(size_t orbital) {
    return BosonExpression(boson::density_density(species, orbital, species, orbital)) -
           BosonExpression(boson::number_op(species, orbital));
  }

  double t_base;
  double t_tooth;
  double u;
  size_t num_cells;
  size_t num_sites;
  Index index;
};
