#pragma once

#include <cstddef>

#include "models/model.h"
#include "term.h"

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
