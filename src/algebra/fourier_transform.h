#pragma once

#include <cmath>
#include <complex>
#include <cstddef>
#include <numbers>
#include <type_traits>
#include <utility>

#include "utils/index.h"

enum class FourierMode { Direct, Inverse };

template <typename OpType, typename F, typename... Args>
using is_operator_callable = std::is_invocable<F, OpType, Args...>;

template <typename F, typename OpType, typename... Args>
auto transform_operator(F&& f, OpType op, Args&&... args)
    -> std::enable_if_t<is_operator_callable<OpType, F, Args...>::value,
                        decltype(f(op, std::forward<Args>(args)...))> {
  return f(op, std::forward<Args>(args)...);
}

template <typename F, typename ExprType, typename... Args>
auto transform_term(F&& f, const ExprType& term_as_expr, Args&&... args)
    -> std::enable_if_t<
        is_operator_callable<typename ExprType::operator_type, F, Args...>::value, ExprType> {
  using monomial_type = typename ExprType::monomial_type;
  // Reconstruct a single-term expression for each operator product
  for (const auto& [ops, coeff] : term_as_expr.terms()) {
    ExprType result(coeff);
    for (const auto& op : ops) {
      result *= transform_operator(std::forward<F>(f), op, std::forward<Args>(args)...);
    }
    return result;
  }
  return ExprType{};
}

template <typename F, typename ExprType, typename... Args>
auto transform_expression(F&& f, const ExprType& expr, Args&&... args)
    -> std::enable_if_t<
        is_operator_callable<typename ExprType::operator_type, F, Args...>::value, ExprType> {
  ExprType result;
  for (const auto& [ops, coeff] : expr.terms()) {
    ExprType term_result(coeff);
    for (const auto& op : ops) {
      term_result *= transform_operator(std::forward<F>(f), op, std::forward<Args>(args)...);
    }
    result += term_result;
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

template <typename ExprType>
ExprType fourier_transform_operator(typename ExprType::operator_type op, const Index& index,
                                    FourierMode mode = FourierMode::Direct) {
  using op_type = typename ExprType::operator_type;
  using monomial_type = typename ExprType::monomial_type;
  using complex_type = typename ExprType::complex_type;

  ExprType result;
  const double type_sign = (op.type() == op_type::Type::Annihilation) ? -1.0 : 1.0;
  const auto from = index(op.value());
  const auto& dimensions = index.dimensions();
  const double normalization = 1.0 / std::sqrt(static_cast<double>(index.size()));
  const double phase_sign = (mode == FourierMode::Direct) ? -1.0 : 1.0;

  for (size_t i = 0; i < index.size(); ++i) {
    const auto to = index(i);
    const auto& orbital = (mode == FourierMode::Direct) ? from : to;
    const auto& momentum = (mode == FourierMode::Direct) ? to : from;
    const double phase = momentum_phase(orbital, momentum, dimensions);
    std::complex<double> coefficient(0.0, phase_sign * type_sign * phase);
    coefficient = std::exp(coefficient) * normalization;

    op_type transformed_op(op.type(), op.spin(), i);
    result += monomial_type(complex_type(coefficient.real(), coefficient.imag()),
                            {transformed_op});
  }

  return result;
}
