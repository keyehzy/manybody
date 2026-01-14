#pragma once

#include <cmath>
#include <complex>
#include <cstddef>
#include <numbers>
#include <type_traits>
#include <utility>

#include "algebra/expression.h"
#include "utils/index.h"

template <typename F, typename... Args>
using is_operator_callable = std::is_invocable<F, Operator, Args...>;

template <typename F, typename... Args>
auto transform_operator(F&& f, Operator op, Args&&... args)
    -> std::enable_if_t<is_operator_callable<F, Args...>::value,
                        decltype(f(op, std::forward<Args>(args)...))> {
  return f(op, std::forward<Args>(args)...);
}

template <typename F, typename... Args>
auto transform_term(F&& f, const Term& term, Args&&... args)
    -> std::enable_if_t<is_operator_callable<F, Args...>::value, Expression> {
  Expression result(term.c);
  for (const auto& op : term.operators) {
    result *= transform_operator(std::forward<F>(f), op, std::forward<Args>(args)...);
  }
  return result;
}

template <typename F, typename... Args>
auto transform_expression(F&& f, const Expression& expr, Args&&... args)
    -> std::enable_if_t<is_operator_callable<F, Args...>::value, Expression> {
  Expression result;
  for (const auto& [ops, coeff] : expr.hashmap) {
    result += transform_term(std::forward<F>(f), Term(coeff, ops), std::forward<Args>(args)...);
  }
  return result;
}

inline double momentum_phase(const Index::container_type& orbital,
                             const Index::container_type& momentum,
                             const Index::container_type& dimensions) {
  double phase = 0.0;
  for (size_t i = 0; i < dimensions.size(); ++i) {
    phase += (static_cast<double>(orbital[i]) * static_cast<double>(momentum[i])) /
             static_cast<double>(dimensions[i]);
  }
  return 2.0 * std::numbers::pi_v<double> * phase;
}

inline Expression fourier_transform_operator(Operator op, const Index& index) {
  Expression result;
  const double type_sign = (op.type() == Operator::Type::Annihilation) ? -1.0 : 1.0;
  const auto orbital = index(op.value());
  const auto& dimensions = index.dimensions();
  const double normalization = 1.0 / std::sqrt(static_cast<double>(index.size()));

  for (size_t k = 0; k < index.size(); ++k) {
    const auto momentum = index(k);
    const double phase = momentum_phase(orbital, momentum, dimensions);
    std::complex<double> coefficient(0.0, -type_sign * phase);
    coefficient = std::exp(coefficient) * normalization;

    Operator transformed_op(op.type(), op.spin(), k);
    result += Term(Term::complex_type(coefficient.real(), coefficient.imag()), {transformed_op});
  }

  return result;
}
