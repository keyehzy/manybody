#pragma once

#include "algebra/fermion/expression.h"

struct Model {
  virtual ~Model() = default;

  virtual FermionExpression hamiltonian() const = 0;
};
