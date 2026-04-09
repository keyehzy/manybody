#include <catch2/catch.hpp>
#include <cmath>
#include <complex>
#include <numbers>

#include "algebra/fourier_transform.h"
#include "algebra/model/hubbard_model.h"
#include "algebra/model/hubbard_model_momentum.h"
#include "utils/index.h"

namespace {
constexpr float kTolerance = 1e-6f;

bool complex_near(const FermionMonomial::complex_type& lhs,
                  const FermionMonomial::complex_type& rhs, float tol) {
  const auto delta = lhs - rhs;
  return std::abs(delta) <= tol;
}

FermionMonomial::complex_type expected_coefficient(FermionOperator op, const Index& index,
                                                   const Index::container_type& other,
                                                   FourierMode mode) {
  const auto from = index(op.value());
  const auto& orbital = (mode == FourierMode::Direct) ? from : other;
  const auto& momentum = (mode == FourierMode::Direct) ? other : from;
  const auto& dimensions = index.dimensions();
  double phase = 0.0;
  for (size_t i = 0; i < dimensions.size(); ++i) {
    phase += (static_cast<double>(orbital[i]) * static_cast<double>(momentum[i])) /
             static_cast<double>(dimensions[i]);
  }
  phase *= 2.0 * std::numbers::pi_v<double>;

  const double type_sign = (op.type() == FermionOperator::Type::Annihilation) ? -1.0 : 1.0;
  const double phase_sign = (mode == FourierMode::Direct) ? -1.0 : 1.0;
  std::complex<double> coefficient(0.0, phase_sign * type_sign * phase);
  coefficient = std::exp(coefficient) / std::sqrt(static_cast<double>(index.size()));
  return FermionMonomial::complex_type(static_cast<float>(coefficient.real()),
                                       static_cast<float>(coefficient.imag()));
}
}  // namespace

TEST_CASE("fourier_transform_operator_multidimensional_coefficients") {
  Index index({2, 3, 2});
  const auto orbital = index({1, 2, 0});
  const auto momentum = Index::container_type{1, 0, 1};
  const auto momentum_orbital = index(momentum);

  FermionOperator annihilation = FermionOperator::annihilation(FermionOperator::Spin::Up, orbital);
  FermionOperator creation = FermionOperator::creation(FermionOperator::Spin::Up, orbital);

  FermionExpression annihilation_expr =
      fourier_transform_operator<FermionExpression>(annihilation, index);
  FermionExpression creation_expr = fourier_transform_operator<FermionExpression>(creation, index);

  FermionExpression::container_type annihilation_ops{
      FermionOperator::annihilation(FermionOperator::Spin::Up, momentum_orbital)};
  FermionExpression::container_type creation_ops{
      FermionOperator::creation(FermionOperator::Spin::Up, momentum_orbital)};

  auto annihilation_it = annihilation_expr.terms().find(annihilation_ops);
  auto creation_it = creation_expr.terms().find(creation_ops);
  CHECK(annihilation_it != annihilation_expr.terms().end());
  CHECK(creation_it != creation_expr.terms().end());

  const auto expected_annihilation =
      expected_coefficient(annihilation, index, momentum, FourierMode::Direct);
  const auto expected_creation = std::conj(expected_annihilation);

  CHECK(complex_near(annihilation_it->second, expected_annihilation, kTolerance));
  CHECK(complex_near(creation_it->second, expected_creation, kTolerance));
}

TEST_CASE("fourier_transform_operator_inverse_multidimensional_coefficients") {
  Index index({2, 3, 2});
  const auto orbital = index({1, 2, 0});
  const auto orbital_coords = index(orbital);
  const auto momentum = Index::container_type{1, 0, 1};
  const auto momentum_orbital = index(momentum);

  FermionOperator annihilation =
      FermionOperator::annihilation(FermionOperator::Spin::Up, momentum_orbital);
  FermionOperator creation = FermionOperator::creation(FermionOperator::Spin::Up, momentum_orbital);

  FermionExpression annihilation_expr =
      fourier_transform_operator<FermionExpression>(annihilation, index, FourierMode::Inverse);
  FermionExpression creation_expr =
      fourier_transform_operator<FermionExpression>(creation, index, FourierMode::Inverse);

  FermionExpression::container_type annihilation_ops{
      FermionOperator::annihilation(FermionOperator::Spin::Up, orbital)};
  FermionExpression::container_type creation_ops{
      FermionOperator::creation(FermionOperator::Spin::Up, orbital)};

  auto annihilation_it = annihilation_expr.terms().find(annihilation_ops);
  auto creation_it = creation_expr.terms().find(creation_ops);
  CHECK(annihilation_it != annihilation_expr.terms().end());
  CHECK(creation_it != creation_expr.terms().end());

  const auto expected_annihilation =
      expected_coefficient(annihilation, index, orbital_coords, FourierMode::Inverse);
  const auto expected_creation = std::conj(expected_annihilation);

  CHECK(complex_near(annihilation_it->second, expected_annihilation, kTolerance));
  CHECK(complex_near(creation_it->second, expected_creation, kTolerance));
}

TEST_CASE("fourier_transform_operator_round_trip_recovers_operator") {
  Index index({4});
  const auto orbital = index({2});

  FermionOperator annihilation =
      FermionOperator::annihilation(FermionOperator::Spin::Down, orbital);
  FermionExpression expr(annihilation);

  FermionExpression momentum = transform_expression(fourier_transform_operator<FermionExpression>,
                                                    expr, index, FourierMode::Direct);
  FermionExpression restored = transform_expression(fourier_transform_operator<FermionExpression>,
                                                    momentum, index, FourierMode::Inverse);

  FermionExpression::container_type ops{annihilation};
  auto it = restored.terms().find(ops);
  REQUIRE(it != restored.terms().end());
  CHECK(complex_near(it->second, FermionMonomial::complex_type{1.0f, 0.0f}, kTolerance));
  CHECK(restored.terms().size() == 1);
}

TEST_CASE("fourier_transform_hubbard_1d_gives_momentum_space") {
  constexpr size_t L = 4;
  constexpr double t = 1.25;
  constexpr double U = 3.5;

  // Real-space model and Fourier transform
  // Note: Real-space hopping with -t gives dispersion -2t*cos(k),
  // matching the momentum-space model.
  HubbardModel hubbard_real(t, U, L);
  Index index({L});

  FermionExpression H_real = hubbard_real.hamiltonian();
  FermionExpression H_transformed = transform_expression(
      fourier_transform_operator<FermionExpression>, H_real, index, FourierMode::Direct);

  HubbardModelMomentum hubbard_momentum(t, U, {L});
  FermionExpression H_momentum = hubbard_momentum.hamiltonian();

  // Normal order both expressions before comparing
  H_transformed = canonicalize(H_transformed);
  H_momentum = canonicalize(H_momentum);

  // Compare expressions term by term
  CHECK(H_transformed.terms().size() == H_momentum.terms().size());
  for (const auto& [ops, coeff] : H_momentum.terms()) {
    auto it = H_transformed.terms().find(ops);
    REQUIRE(it != H_transformed.terms().end());
    CHECK(complex_near(it->second, coeff, kTolerance));
  }
}

TEST_CASE("fourier_transform_hubbard_2d_gives_momentum_space") {
  constexpr size_t Lx = 2;
  constexpr size_t Ly = 3;
  constexpr double t = 0.75;
  constexpr double U = 2.0;

  HubbardModel2D hubbard_real(t, U, Lx, Ly);
  Index index({Lx, Ly});

  FermionExpression H_real = hubbard_real.hamiltonian();
  FermionExpression H_transformed = transform_expression(
      fourier_transform_operator<FermionExpression>, H_real, index, FourierMode::Direct);

  HubbardModelMomentum hubbard_momentum(t, U, {Lx, Ly});
  FermionExpression H_momentum = hubbard_momentum.hamiltonian();

  // Normal order both expressions before comparing
  H_transformed = canonicalize(H_transformed);
  H_momentum = canonicalize(H_momentum);

  CHECK(H_transformed.terms().size() == H_momentum.terms().size());
  for (const auto& [ops, coeff] : H_momentum.terms()) {
    auto it = H_transformed.terms().find(ops);
    REQUIRE(it != H_transformed.terms().end());
    CHECK(complex_near(it->second, coeff, kTolerance));
  }
}

TEST_CASE("fourier_transform_hubbard_3d_gives_momentum_space") {
  constexpr size_t Lx = 2;
  constexpr size_t Ly = 2;
  constexpr size_t Lz = 2;
  constexpr double t = 1.0;
  constexpr double U = 4.0;

  HubbardModel3D hubbard_real(t, U, Lx, Ly, Lz);
  Index index({Lx, Ly, Lz});

  FermionExpression H_real = hubbard_real.hamiltonian();
  FermionExpression H_transformed = transform_expression(
      fourier_transform_operator<FermionExpression>, H_real, index, FourierMode::Direct);

  HubbardModelMomentum hubbard_momentum(t, U, {Lx, Ly, Lz});
  FermionExpression H_momentum = hubbard_momentum.hamiltonian();

  // Normal order both expressions before comparing
  H_transformed = canonicalize(H_transformed);
  H_momentum = canonicalize(H_momentum);

  CHECK(H_transformed.terms().size() == H_momentum.terms().size());
  for (const auto& [ops, coeff] : H_momentum.terms()) {
    auto it = H_transformed.terms().find(ops);
    REQUIRE(it != H_transformed.terms().end());
    CHECK(complex_near(it->second, coeff, kTolerance));
  }
}
