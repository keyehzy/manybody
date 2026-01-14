#include <cmath>
#include <complex>
#include <numbers>

#include "algebra/fourier_transform.h"
#include "catch.hpp"
#include "utils/index.h"

namespace {
constexpr double kTolerance = 1e-6;

bool complex_near(const Term::complex_type& lhs, const Term::complex_type& rhs, double tol) {
  const auto delta = lhs - rhs;
  return std::abs(delta) <= tol;
}

Term::complex_type expected_coefficient(Operator op, const Index& index,
                                        const Index::container_type& momentum) {
  const auto orbital = index(op.value());
  const auto& dimensions = index.dimensions();
  double phase = 0.0;
  for (size_t i = 0; i < dimensions.size(); ++i) {
    phase += (static_cast<double>(orbital[i]) * static_cast<double>(momentum[i])) /
             static_cast<double>(dimensions[i]);
  }
  phase *= 2.0 * std::numbers::pi_v<double>;

  const double type_sign = (op.type() == Operator::Type::Annihilation) ? -1.0 : 1.0;
  std::complex<double> coefficient(0.0, -type_sign * phase);
  coefficient = std::exp(coefficient) / std::sqrt(static_cast<double>(index.size()));
  return Term::complex_type(static_cast<float>(coefficient.real()),
                            static_cast<float>(coefficient.imag()));
}
}  // namespace

TEST_CASE("fourier_transform_operator_multidimensional_coefficients") {
  Index index({2, 3, 2});
  const auto orbital = index({1, 2, 0});
  const auto momentum = Index::container_type{1, 0, 1};
  const auto momentum_orbital = index(momentum);

  Operator annihilation = Operator::annihilation(Operator::Spin::Up, orbital);
  Operator creation = Operator::creation(Operator::Spin::Up, orbital);

  Expression annihilation_expr = fourier_transform_operator(annihilation, index);
  Expression creation_expr = fourier_transform_operator(creation, index);

  Expression::container_type annihilation_ops{
      Operator::annihilation(Operator::Spin::Up, momentum_orbital)};
  Expression::container_type creation_ops{Operator::creation(Operator::Spin::Up, momentum_orbital)};

  auto annihilation_it = annihilation_expr.hashmap.find(annihilation_ops);
  auto creation_it = creation_expr.hashmap.find(creation_ops);
  CHECK(annihilation_it != annihilation_expr.hashmap.end());
  CHECK(creation_it != creation_expr.hashmap.end());

  const auto expected_annihilation = expected_coefficient(annihilation, index, momentum);
  const auto expected_creation = std::conj(expected_annihilation);

  CHECK(complex_near(annihilation_it->second, expected_annihilation, kTolerance));
  CHECK(complex_near(creation_it->second, expected_creation, kTolerance));
}
