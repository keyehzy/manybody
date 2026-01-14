#pragma once

#include <cstddef>
#include <vector>

#include "algebra/model/model.h"
#include "algebra/term.h"
#include "utils/index.h"

struct HubbardModel : Model {
  HubbardModel(double t, double u, size_t size) : t(t), u(u), size(size) {}

  Expression kinetic() const {
    Expression kinetic_term;
    for (size_t i = 0; i < size; ++i) {
      const size_t next = (i + 1) % size;
      kinetic_term += t * hopping(i, next, Operator::Spin::Up);
      kinetic_term += t * hopping(i, next, Operator::Spin::Down);
    }
    return kinetic_term;
  }

  Expression interaction() const {
    Expression interaction_term;
    for (size_t i = 0; i < size; ++i) {
      interaction_term += u * density_density(Operator::Spin::Up, i, Operator::Spin::Down, i);
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
  HubbardModel2D(double t, double u, size_t size_x, size_t size_y)
      : t(t), u(u), size_x(size_x), size_y(size_y), index({size_x, size_y}) {}

  Expression kinetic() const {
    Expression kinetic_term;
    for (size_t x = 0; x < size_x; ++x) {
      for (size_t y = 0; y < size_y; ++y) {
        const size_t site = index({x, y});
        const size_t x_next = index({(x + 1) % size_x, y});
        const size_t y_next = index({x, (y + 1) % size_y});
        kinetic_term += t * hopping(site, x_next, Operator::Spin::Up);
        kinetic_term += t * hopping(site, x_next, Operator::Spin::Down);
        kinetic_term += t * hopping(site, y_next, Operator::Spin::Up);
        kinetic_term += t * hopping(site, y_next, Operator::Spin::Down);
      }
    }
    return kinetic_term;
  }

  Expression interaction() const {
    Expression interaction_term;
    for (size_t x = 0; x < size_x; ++x) {
      for (size_t y = 0; y < size_y; ++y) {
        const size_t site = index({x, y});
        interaction_term +=
            u * density_density(Operator::Spin::Up, site, Operator::Spin::Down, site);
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
  HubbardModel3D(double t, double u, size_t size_x, size_t size_y, size_t size_z)
      : t(t),
        u(u),
        size_x(size_x),
        size_y(size_y),
        size_z(size_z),
        index({size_x, size_y, size_z}) {}

  Expression kinetic() const {
    Expression kinetic_term;
    for (size_t x = 0; x < size_x; ++x) {
      for (size_t y = 0; y < size_y; ++y) {
        for (size_t z = 0; z < size_z; ++z) {
          const size_t site = index({x, y, z});
          const size_t x_next = index({(x + 1) % size_x, y, z});
          const size_t y_next = index({x, (y + 1) % size_y, z});
          const size_t z_next = index({x, y, (z + 1) % size_z});
          kinetic_term += t * hopping(site, x_next, Operator::Spin::Up);
          kinetic_term += t * hopping(site, x_next, Operator::Spin::Down);
          kinetic_term += t * hopping(site, y_next, Operator::Spin::Up);
          kinetic_term += t * hopping(site, y_next, Operator::Spin::Down);
          kinetic_term += t * hopping(site, z_next, Operator::Spin::Up);
          kinetic_term += t * hopping(site, z_next, Operator::Spin::Down);
        }
      }
    }
    return kinetic_term;
  }

  Expression interaction() const {
    Expression interaction_term;
    for (size_t x = 0; x < size_x; ++x) {
      for (size_t y = 0; y < size_y; ++y) {
        for (size_t z = 0; z < size_z; ++z) {
          const size_t site = index({x, y, z});
          interaction_term +=
              u * density_density(Operator::Spin::Up, site, Operator::Spin::Down, site);
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
