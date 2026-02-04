#include "algebra/expression.h"

#include <algorithm>
#include <sstream>
#include <utility>

FermionExpression FermionExpression::adjoint() const {
  FermionExpression result;
  for (const auto& [ops, coeff] : this->data) {
    FermionMonomial term(coeff, ops);
    FermionMonomial adj = ::adjoint(term);
    result.add_to_map(std::move(adj.operators), adj.c);
  }
  return result;
}

FermionExpression& FermionExpression::truncate_by_size(size_t max_size) {
  if (max_size == 0) {
    this->clear();
    return *this;
  }
  for (auto it = this->data.begin(); it != this->data.end();) {
    if (it->first.size() > max_size) {
      it = this->data.erase(it);
    } else {
      ++it;
    }
  }
  return *this;
}

FermionExpression& FermionExpression::filter_by_size(size_t size) {
  if (size == 0) {
    this->clear();
    return *this;
  }
  for (auto it = this->data.begin(); it != this->data.end();) {
    if (it->first.size() != size) {
      it = this->data.erase(it);
    } else {
      ++it;
    }
  }
  return *this;
}

void FermionExpression::format_to(std::ostringstream& oss) const {
  this->format_sorted(
      oss, [](std::ostringstream& os, const container_type& ops, const complex_type& coeff) {
        FermionMonomial term(coeff, ops);
        ::to_string(os, term);
      });
}

std::string FermionExpression::to_string() const {
  return Base::to_string(
      [](std::ostringstream& os, const container_type& ops, const complex_type& coeff) {
        FermionMonomial term(coeff, ops);
        ::to_string(os, term);
      });
}

FermionExpression& FermionExpression::operator*=(const FermionExpression& value) {
  if (this->empty() || value.empty()) {
    this->clear();
    return *this;
  }
  map_type result;
  result.reserve(this->size() * value.size());
  for (const auto& [lhs_ops, lhs_coeff] : this->data) {
    for (const auto& [rhs_ops, rhs_coeff] : value.data) {
      if (lhs_ops.size() + rhs_ops.size() > FermionMonomial::container_type::max_size()) {
        continue;
      }
      container_type combined = lhs_ops;
      combined.append_range(rhs_ops.begin(), rhs_ops.end());
      add_to(result, std::move(combined), lhs_coeff * rhs_coeff);
    }
  }
  this->data = std::move(result);
  return *this;
}

FermionExpression& FermionExpression::operator*=(const FermionMonomial& value) {
  if (this->empty()) {
    return *this;
  }
  if (this->is_zero(value.c)) {
    this->clear();
    return *this;
  }

  if (value.operators.size() == 0) {
    for (auto& [ops, coeff] : this->data) {
      coeff *= value.c;
    }
    return *this;
  }
  map_type result;
  result.reserve(this->size());
  for (const auto& [ops, coeff] : this->data) {
    if (ops.size() + value.operators.size() > FermionMonomial::container_type::max_size()) {
      continue;
    }
    container_type combined = ops;
    combined.append_range(value.operators.begin(), value.operators.end());
    add_to(result, std::move(combined), coeff * value.c);
  }
  this->data = std::move(result);
  return *this;
}
