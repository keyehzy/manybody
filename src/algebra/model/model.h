#pragma once

#include "algebra/expression.h"

struct Model {
  virtual ~Model() = default;

  virtual Expression hamiltonian() const = 0;
};
