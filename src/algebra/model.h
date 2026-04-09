#pragma once

template <typename ExpressionType>
struct BasicModel {
  using expression_type = ExpressionType;

  virtual ~BasicModel() = default;

  virtual expression_type hamiltonian() const = 0;
};
