#pragma once

#include "algebra/expression.h"
#include "robin_hood.h"

struct NormalOrderer {
  using complex_type = FermionMonomial::complex_type;
  using container_type = FermionMonomial::container_type;

  NormalOrderer() = default;

  Expression normal_order(const complex_type& c, const container_type& ops);
  Expression normal_order(const FermionMonomial& term);
  Expression normal_order(const Expression& expr);

  Expression normal_order_recursive(container_type ops);
  Expression normal_order_recursive(container_type ops, std::size_t ops_hash);
  Expression handle_non_commuting(container_type ops, size_t index);
};
