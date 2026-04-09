#pragma once

#include <cstddef>
#include <vector>

#include "algebra/fermion/model/model.h"
#include "algebra/fermion/term.h"
#include "utils/index.h"

struct HubbardModel : FermionModel {
  HubbardModel(double t_val, double u_val, size_t size_val) : t(t_val), u(u_val), size(size_val) {}

  FermionExpression kinetic() const {
    FermionExpression kinetic_term;
    const auto t_coeff = FermionExpression::complex_type(-t, 0.0);
    for (size_t i = 0; i < size; ++i) {
      const size_t next = (i + 1) % size;
      kinetic_term += t_coeff * hopping(i, next, FermionOperator::Spin::Up);
      kinetic_term += t_coeff * hopping(i, next, FermionOperator::Spin::Down);
    }
    return kinetic_term;
  }

  FermionExpression interaction() const {
    FermionExpression interaction_term;
    const auto u_coeff = FermionExpression::complex_type(u, 0.0);
    for (size_t i = 0; i < size; ++i) {
      interaction_term +=
          u_coeff * density_density(FermionOperator::Spin::Up, i, FermionOperator::Spin::Down, i);
    }
    return interaction_term;
  }

  FermionExpression hamiltonian() const override {
    FermionExpression result = kinetic();
    result += interaction();
    return result;
  }

  /// Opposite-spin density-density correlation operator G_{↑↓}(r):
  /// G_{↑↓}(r) = (1/N) Σ_i n_{i,↑} n_{i+r,↓}
  ///
  /// Measures spatial extent of opposite-spin correlations.
  /// If r=0, this measures on-site pairs; for r≠0 it measures extended pairs.
  FermionExpression opposite_spin_correlation(size_t r) const {
    FermionExpression result;
    const auto coeff = FermionExpression::complex_type(1.0 / static_cast<double>(size), 0.0);
    for (size_t i = 0; i < size; ++i) {
      const size_t i_plus_r = (i + r) % size;
      result += coeff * density_density(FermionOperator::Spin::Up, i, FermionOperator::Spin::Down,
                                        i_plus_r);
    }
    return result;
  }

  double t;
  double u;
  size_t size;
};

struct HubbardModel2D : FermionModel {
  HubbardModel2D(double t_val, double u_val, size_t size_x_val, size_t size_y_val)
      : t(t_val),
        u(u_val),
        size_x(size_x_val),
        size_y(size_y_val),
        index({size_x_val, size_y_val}) {}

  FermionExpression kinetic() const {
    FermionExpression kinetic_term;
    const auto t_coeff = FermionExpression::complex_type(-t, 0.0);
    for (size_t x = 0; x < size_x; ++x) {
      for (size_t y = 0; y < size_y; ++y) {
        const size_t site = index({x, y});
        const size_t x_next = index({(x + 1) % size_x, y});
        const size_t y_next = index({x, (y + 1) % size_y});
        kinetic_term += t_coeff * hopping(site, x_next, FermionOperator::Spin::Up);
        kinetic_term += t_coeff * hopping(site, x_next, FermionOperator::Spin::Down);
        kinetic_term += t_coeff * hopping(site, y_next, FermionOperator::Spin::Up);
        kinetic_term += t_coeff * hopping(site, y_next, FermionOperator::Spin::Down);
      }
    }
    return kinetic_term;
  }

  FermionExpression interaction() const {
    FermionExpression interaction_term;
    const auto u_coeff = FermionExpression::complex_type(u, 0.0);
    for (size_t x = 0; x < size_x; ++x) {
      for (size_t y = 0; y < size_y; ++y) {
        const size_t site = index({x, y});
        interaction_term += u_coeff * density_density(FermionOperator::Spin::Up, site,
                                                      FermionOperator::Spin::Down, site);
      }
    }
    return interaction_term;
  }

  FermionExpression hamiltonian() const override {
    FermionExpression result = kinetic();
    result += interaction();
    return result;
  }

  /// Opposite-spin density-density correlation operator G_{↑↓}(r):
  /// G_{↑↓}(r) = (1/N) Σ_i n_{i,↑} n_{i+r,↓}
  ///
  /// Measures spatial extent of opposite-spin correlations.
  FermionExpression opposite_spin_correlation(size_t rx, size_t ry) const {
    FermionExpression result;
    const size_t N = size_x * size_y;
    const auto coeff = FermionExpression::complex_type(1.0 / static_cast<double>(N), 0.0);
    for (size_t x = 0; x < size_x; ++x) {
      for (size_t y = 0; y < size_y; ++y) {
        const size_t site_i = index({x, y});
        const size_t site_i_plus_r = index({(x + rx) % size_x, (y + ry) % size_y});
        result += coeff * density_density(FermionOperator::Spin::Up, site_i,
                                          FermionOperator::Spin::Down, site_i_plus_r);
      }
    }
    return result;
  }

  double t;
  double u;
  size_t size_x;
  size_t size_y;
  Index index;
};

struct HubbardModel3D : FermionModel {
  HubbardModel3D(double t_val, double u_val, size_t size_x_val, size_t size_y_val,
                 size_t size_z_val)
      : t(t_val),
        u(u_val),
        size_x(size_x_val),
        size_y(size_y_val),
        size_z(size_z_val),
        index({size_x_val, size_y_val, size_z_val}) {}

  FermionExpression kinetic() const {
    FermionExpression kinetic_term;
    const auto t_coeff = FermionExpression::complex_type(-t, 0.0);
    for (size_t x = 0; x < size_x; ++x) {
      for (size_t y = 0; y < size_y; ++y) {
        for (size_t z = 0; z < size_z; ++z) {
          const size_t site = index({x, y, z});
          const size_t x_next = index({(x + 1) % size_x, y, z});
          const size_t y_next = index({x, (y + 1) % size_y, z});
          const size_t z_next = index({x, y, (z + 1) % size_z});
          kinetic_term += t_coeff * hopping(site, x_next, FermionOperator::Spin::Up);
          kinetic_term += t_coeff * hopping(site, x_next, FermionOperator::Spin::Down);
          kinetic_term += t_coeff * hopping(site, y_next, FermionOperator::Spin::Up);
          kinetic_term += t_coeff * hopping(site, y_next, FermionOperator::Spin::Down);
          kinetic_term += t_coeff * hopping(site, z_next, FermionOperator::Spin::Up);
          kinetic_term += t_coeff * hopping(site, z_next, FermionOperator::Spin::Down);
        }
      }
    }
    return kinetic_term;
  }

  FermionExpression interaction() const {
    FermionExpression interaction_term;
    const auto u_coeff = FermionExpression::complex_type(u, 0.0);
    for (size_t x = 0; x < size_x; ++x) {
      for (size_t y = 0; y < size_y; ++y) {
        for (size_t z = 0; z < size_z; ++z) {
          const size_t site = index({x, y, z});
          interaction_term += u_coeff * density_density(FermionOperator::Spin::Up, site,
                                                        FermionOperator::Spin::Down, site);
        }
      }
    }
    return interaction_term;
  }

  FermionExpression hamiltonian() const override {
    FermionExpression result = kinetic();
    result += interaction();
    return result;
  }

  /// Opposite-spin density-density correlation operator G_{↑↓}(r):
  /// G_{↑↓}(r) = (1/N) Σ_i n_{i,↑} n_{i+r,↓}
  ///
  /// Measures spatial extent of opposite-spin correlations.
  FermionExpression opposite_spin_correlation(size_t rx, size_t ry, size_t rz) const {
    FermionExpression result;
    const size_t N = size_x * size_y * size_z;
    const auto coeff = FermionExpression::complex_type(1.0 / static_cast<double>(N), 0.0);
    for (size_t x = 0; x < size_x; ++x) {
      for (size_t y = 0; y < size_y; ++y) {
        for (size_t z = 0; z < size_z; ++z) {
          const size_t site_i = index({x, y, z});
          const size_t site_i_plus_r =
              index({(x + rx) % size_x, (y + ry) % size_y, (z + rz) % size_z});
          result += coeff * density_density(FermionOperator::Spin::Up, site_i,
                                            FermionOperator::Spin::Down, site_i_plus_r);
        }
      }
    }
    return result;
  }

  double t;
  double u;
  size_t size_x;
  size_t size_y;
  size_t size_z;
  Index index;
};
