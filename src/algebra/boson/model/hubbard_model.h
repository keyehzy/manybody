#pragma once

#include <cstddef>
#include <stdexcept>

#include "algebra/boson/expression.h"
#include "algebra/model.h"
#include "utils/index.h"

/// Single-component Bose-Hubbard model on a 2D square lattice with periodic boundaries.
///
/// H = -t sum_<ij> (b_i^dag b_j + b_j^dag b_i)
///   + U/2 sum_i n_i (n_i - 1).
///
/// The spin degree of freedom is repurposed as a species label, with all particles using Spin::Up.
struct BoseHubbardModel2D : BasicModel<BosonExpression> {
  static constexpr BosonOperator::Spin species = BosonOperator::Spin::Up;

  BoseHubbardModel2D(double t_val, double u_val, size_t size_x_val, size_t size_y_val)
      : t(t_val),
        u(u_val),
        size_x(size_x_val),
        size_y(size_y_val),
        num_sites(size_x_val * size_y_val),
        index(checked_dimensions(size_x_val, size_y_val)) {}

  size_t site(size_t x, size_t y) const { return index({x, y}); }

  BosonExpression kinetic() const {
    BosonExpression result;
    const auto coeff = BosonExpression::complex_type(-t, 0.0);

    for (size_t x = 0; x < size_x; ++x) {
      for (size_t y = 0; y < size_y; ++y) {
        const size_t src = site(x, y);
        const size_t x_next = site((x + 1) % size_x, y);
        const size_t y_next = site(x, (y + 1) % size_y);

        if (x_next != src) {
          result += hopping(coeff, src, x_next, species);
        }
        if (y_next != src) {
          result += hopping(coeff, src, y_next, species);
        }
      }
    }

    return result;
  }

  BosonExpression interaction() const {
    BosonExpression result;
    const auto coeff = BosonExpression::complex_type(0.5 * u, 0.0);

    for (size_t site_index = 0; site_index < num_sites; ++site_index) {
      result += coeff * onsite_interaction(site_index);
    }

    return result;
  }

  BosonExpression hamiltonian() const override {
    BosonExpression result = kinetic();
    result += interaction();
    return result;
  }

  BosonExpression current(size_t direction) const {
    if (direction >= 2) {
      throw std::invalid_argument("BoseHubbardModel2D current direction must be 0 or 1.");
    }

    BosonExpression result;
    const auto coeff = BosonExpression::complex_type(0.0, -t);

    for (size_t x = 0; x < size_x; ++x) {
      for (size_t y = 0; y < size_y; ++y) {
        const size_t src = site(x, y);
        const size_t dst = direction == 0 ? site((x + 1) % size_x, y) : site(x, (y + 1) % size_y);

        if (dst != src) {
          result += hopping(coeff, src, dst, species);
        }
      }
    }

    return result;
  }

  static BosonExpression onsite_interaction(size_t orbital) {
    return BosonExpression(boson::density_density(species, orbital, species, orbital)) -
           BosonExpression(boson::number_op(species, orbital));
  }

  static Index::container_type checked_dimensions(size_t size_x_val, size_t size_y_val) {
    if (size_x_val == 0 || size_y_val == 0) {
      throw std::invalid_argument("BoseHubbardModel2D requires positive lattice dimensions.");
    }
    return {size_x_val, size_y_val};
  }

  double t;
  double u;
  size_t size_x;
  size_t size_y;
  size_t num_sites;
  Index index;
};
