#pragma once

#include "expression.h"

struct Model {
  virtual ~Model() = default;

  virtual Expression hamiltonian() const = 0;
};
