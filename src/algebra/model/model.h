#pragma once

#include "algebra/fermion/expression.h"

struct Model {
  virtual ~Model() = default;

  virtual Expression hamiltonian() const = 0;
};
