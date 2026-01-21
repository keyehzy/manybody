#pragma once

#include <cstddef>
#include <vector>

#include "algebra/model/model.h"
#include "algebra/term.h"
#include "utils/index.h"

struct HubbardModel : Model {
  HubbardModel(double t_val, double u_val, size_t size_val) : t(t_val), u(u_val), size(size_val) {}

  Expression kinetic() const {
    Expression kinetic_term;
    const auto t_coeff = Expression::complex_type(-t, 0.0);
    for (size_t i = 0; i < size; ++i) {
      const size_t next = (i + 1) % size;
      kinetic_term += t_coeff * hopping(i, next, Operator::Spin::Up);
      kinetic_term += t_coeff * hopping(i, next, Operator::Spin::Down);
    }
    return kinetic_term;
  }

  Expression interaction() const {
    Expression interaction_term;
    const auto u_coeff = Expression::complex_type(u, 0.0);
    for (size_t i = 0; i < size; ++i) {
      interaction_term += u_coeff * density_density(Operator::Spin::Up, i, Operator::Spin::Down, i);
    }
    return interaction_term;
  }

  Expression hamiltonian() const override {
    Expression result = kinetic();
    result += interaction();
    return result;
  }

  double t;
  double u;
  size_t size;
};

struct HubbardModel2D : Model {
  HubbardModel2D(double t_val, double u_val, size_t size_x_val, size_t size_y_val)
      : t(t_val),
        u(u_val),
        size_x(size_x_val),
        size_y(size_y_val),
        index({size_x_val, size_y_val}) {}

  Expression kinetic() const {
    Expression kinetic_term;
    const auto t_coeff = Expression::complex_type(-t, 0.0);
    for (size_t x = 0; x < size_x; ++x) {
      for (size_t y = 0; y < size_y; ++y) {
        const size_t site = index({x, y});
        const size_t x_next = index({(x + 1) % size_x, y});
        const size_t y_next = index({x, (y + 1) % size_y});
        kinetic_term += t_coeff * hopping(site, x_next, Operator::Spin::Up);
        kinetic_term += t_coeff * hopping(site, x_next, Operator::Spin::Down);
        kinetic_term += t_coeff * hopping(site, y_next, Operator::Spin::Up);
        kinetic_term += t_coeff * hopping(site, y_next, Operator::Spin::Down);
      }
    }
    return kinetic_term;
  }

  Expression interaction() const {
    Expression interaction_term;
    const auto u_coeff = Expression::complex_type(u, 0.0);
    for (size_t x = 0; x < size_x; ++x) {
      for (size_t y = 0; y < size_y; ++y) {
        const size_t site = index({x, y});
        interaction_term +=
            u_coeff * density_density(Operator::Spin::Up, site, Operator::Spin::Down, site);
      }
    }
    return interaction_term;
  }

  Expression hamiltonian() const override {
    Expression result = kinetic();
    result += interaction();
    return result;
  }

  double t;
  double u;
  size_t size_x;
  size_t size_y;
  Index index;
};

struct HubbardModel3D : Model {
  HubbardModel3D(double t_val, double u_val, size_t size_x_val, size_t size_y_val,
                 size_t size_z_val)
      : t(t_val),
        u(u_val),
        size_x(size_x_val),
        size_y(size_y_val),
        size_z(size_z_val),
        index({size_x_val, size_y_val, size_z_val}) {}

  Expression kinetic() const {
    Expression kinetic_term;
    const auto t_coeff = Expression::complex_type(-t, 0.0);
    for (size_t x = 0; x < size_x; ++x) {
      for (size_t y = 0; y < size_y; ++y) {
        for (size_t z = 0; z < size_z; ++z) {
          const size_t site = index({x, y, z});
          const size_t x_next = index({(x + 1) % size_x, y, z});
          const size_t y_next = index({x, (y + 1) % size_y, z});
          const size_t z_next = index({x, y, (z + 1) % size_z});
          kinetic_term += t_coeff * hopping(site, x_next, Operator::Spin::Up);
          kinetic_term += t_coeff * hopping(site, x_next, Operator::Spin::Down);
          kinetic_term += t_coeff * hopping(site, y_next, Operator::Spin::Up);
          kinetic_term += t_coeff * hopping(site, y_next, Operator::Spin::Down);
          kinetic_term += t_coeff * hopping(site, z_next, Operator::Spin::Up);
          kinetic_term += t_coeff * hopping(site, z_next, Operator::Spin::Down);
        }
      }
    }
    return kinetic_term;
  }

  Expression interaction() const {
    Expression interaction_term;
    const auto u_coeff = Expression::complex_type(u, 0.0);
    for (size_t x = 0; x < size_x; ++x) {
      for (size_t y = 0; y < size_y; ++y) {
        for (size_t z = 0; z < size_z; ++z) {
          const size_t site = index({x, y, z});
          interaction_term +=
              u_coeff * density_density(Operator::Spin::Up, site, Operator::Spin::Down, site);
        }
      }
    }
    return interaction_term;
  }

  Expression hamiltonian() const override {
    Expression result = kinetic();
    result += interaction();
    return result;
  }

  double t;
  double u;
  size_t size_x;
  size_t size_y;
  size_t size_z;
  Index index;
};
