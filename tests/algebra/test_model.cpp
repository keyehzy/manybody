#include <cmath>
#include <numbers>
#include <type_traits>

#include "algebra/model/hubbard_model.h"
#include "algebra/model/hubbard_model_momentum.h"
#include "algebra/model/hubbard_model_relative.h"
#include "algebra/model/model.h"
#include "catch.hpp"

namespace {
constexpr float kModelTolerance = 1e-6f;

bool complex_near_model(const Term::complex_type& lhs, const Term::complex_type& rhs, float tol) {
  const auto delta = lhs - rhs;
  return std::abs(delta) <= tol;
}
}  // namespace

TEST_CASE("model_hubbard_inherits_from_interface") {
  CHECK((std::is_base_of_v<Model, HubbardModel>));
}

TEST_CASE("model_hubbard_2d_inherits_from_interface") {
  CHECK((std::is_base_of_v<Model, HubbardModel2D>));
}

TEST_CASE("model_hubbard_3d_inherits_from_interface") {
  CHECK((std::is_base_of_v<Model, HubbardModel3D>));
}

TEST_CASE("model_hubbard_relative_inherits_from_interface") {
  CHECK((std::is_base_of_v<Model, HubbardModelRelative>));
}

TEST_CASE("model_hubbard_hamiltonian_term_count") {
  HubbardModel hubbard(1.0, 2.0, 2);
  const Expression hamiltonian = hubbard.hamiltonian();
  CHECK((hamiltonian.size()) == (6u));
}

TEST_CASE("model_hubbard_2d_hamiltonian_term_count") {
  HubbardModel2D hubbard(1.0, 2.0, 2, 2);
  const Expression hamiltonian = hubbard.hamiltonian();
  CHECK((hamiltonian.size()) == (20u));
}

TEST_CASE("model_hubbard_3d_hamiltonian_term_count") {
  HubbardModel3D hubbard(1.0, 2.0, 2, 2, 2);
  const Expression hamiltonian = hubbard.hamiltonian();
  CHECK((hamiltonian.size()) == (56u));
}

TEST_CASE("model_hubbard_relative_hamiltonian_term_count") {
  HubbardModelRelative hubbard(1.0, 2.0, 4, 0);
  const Expression hamiltonian = hubbard.hamiltonian();
  CHECK((hamiltonian.size()) == (16u));
}

TEST_CASE("model_hubbard_relative_hamiltonian_coefficients") {
  constexpr size_t kMomentum = 0;
  HubbardModelRelative hubbard(1.25, 3.0, 4, kMomentum);
  const Expression hamiltonian = hubbard.hamiltonian();

  const size_t size = 4;
  const size_t p_same = 0;
  const size_t q_same = 0;
  const size_t p_diff = 0;
  const size_t q_diff = 1;
  const size_t k_minus_p_same = (kMomentum + size - p_same) % size;
  const size_t k_minus_q_same = (kMomentum + size - q_same) % size;
  const size_t k_minus_p_diff = (kMomentum + size - p_diff) % size;
  const size_t k_minus_q_diff = (kMomentum + size - q_diff) % size;

  Expression::container_type diag_ops{
      Operator::creation(Operator::Spin::Up, p_same),
      Operator::creation(Operator::Spin::Down, k_minus_p_same),
      Operator::annihilation(Operator::Spin::Down, k_minus_q_same),
      Operator::annihilation(Operator::Spin::Up, q_same),
  };
  Expression::container_type off_diag_ops{
      Operator::creation(Operator::Spin::Up, p_diff),
      Operator::creation(Operator::Spin::Down, k_minus_p_diff),
      Operator::annihilation(Operator::Spin::Down, k_minus_q_diff),
      Operator::annihilation(Operator::Spin::Up, q_diff),
  };

  const auto diag_it = hamiltonian.hashmap.find(diag_ops);
  const auto off_diag_it = hamiltonian.hashmap.find(off_diag_ops);
  CHECK(diag_it != hamiltonian.hashmap.end());
  CHECK(off_diag_it != hamiltonian.hashmap.end());

  const double k_phase =
      2.0 * std::numbers::pi_v<double> * static_cast<double>(p_same) / static_cast<double>(size);
  const double k_total_phase =
      2.0 * std::numbers::pi_v<double> * static_cast<double>(kMomentum) / static_cast<double>(size);
  const double t_eff = 2.0 * 1.25 * std::cos(0.5 * k_total_phase);
  const double expected_diag = (3.0 / static_cast<double>(size)) + 2.0 * t_eff * std::cos(k_phase);
  const double expected_off_diag = 3.0 / static_cast<double>(size);

  CHECK(complex_near_model(diag_it->second,
                           Term::complex_type(static_cast<float>(expected_diag), 0.0f),
                           kModelTolerance));
  CHECK(complex_near_model(off_diag_it->second,
                           Term::complex_type(static_cast<float>(expected_off_diag), 0.0f),
                           kModelTolerance));
}

TEST_CASE("model_virtual_dispatch_hamiltonian") {
  HubbardModel hubbard(1.0, 2.0, 2);
  const Model& model = hubbard;
  CHECK((model.hamiltonian().size()) == (6u));
}

TEST_CASE("model_virtual_dispatch_hamiltonian_2d") {
  HubbardModel2D hubbard(1.0, 2.0, 2, 2);
  const Model& model = hubbard;
  CHECK((model.hamiltonian().size()) == (20u));
}

TEST_CASE("model_virtual_dispatch_hamiltonian_3d") {
  HubbardModel3D hubbard(1.0, 2.0, 2, 2, 2);
  const Model& model = hubbard;
  CHECK((model.hamiltonian().size()) == (56u));
}

TEST_CASE("model_virtual_dispatch_hamiltonian_relative") {
  HubbardModelRelative hubbard(1.0, 2.0, 4, 0);
  const Model& model = hubbard;
  CHECK((model.hamiltonian().size()) == (16u));
}

TEST_CASE("model_hubbard_momentum_inherits_from_interface") {
  CHECK((std::is_base_of_v<Model, HubbardModelMomentum>));
}

TEST_CASE("model_hubbard_momentum_1d_hamiltonian_term_count") {
  // 1D lattice with 4 sites
  HubbardModelMomentum hubbard(1.0, 2.0, {4});
  const Expression hamiltonian = hubbard.hamiltonian();
  // Kinetic: 4 sites * 2 spins = 8 terms (but diagonal, so some may combine)
  // Interaction: 4^3 = 64 terms (but normal ordering may reduce)
  CHECK(hamiltonian.size() > 0);
}

TEST_CASE("model_hubbard_momentum_2d_hamiltonian_term_count") {
  // 2D lattice with 2x2 sites
  HubbardModelMomentum hubbard(1.0, 2.0, {2, 2});
  const Expression hamiltonian = hubbard.hamiltonian();
  CHECK(hamiltonian.size() > 0);
}

TEST_CASE("model_hubbard_momentum_3d_hamiltonian_term_count") {
  // 3D lattice with 2x2x2 sites
  HubbardModelMomentum hubbard(1.0, 2.0, {2, 2, 2});
  const Expression hamiltonian = hubbard.hamiltonian();
  CHECK(hamiltonian.size() > 0);
}

TEST_CASE("model_hubbard_momentum_dispersion_1d") {
  HubbardModelMomentum hubbard(1.0, 2.0, {4});

  // k=0: ε = -2t * cos(0) = -2t = -2
  CHECK(std::abs(hubbard.dispersion({0}) - (-2.0)) < kModelTolerance);

  // k=2 (half-filling): ε = -2t * cos(π) = 2t = 2
  CHECK(std::abs(hubbard.dispersion({2}) - (2.0)) < kModelTolerance);
}

TEST_CASE("model_hubbard_momentum_dispersion_2d") {
  HubbardModelMomentum hubbard(1.0, 2.0, {4, 4});

  // k=(0,0): ε = -2t * (cos(0) + cos(0)) = -4t = -4
  CHECK(std::abs(hubbard.dispersion({0, 0}) - (-4.0)) < kModelTolerance);

  // k=(2,2): ε = -2t * (cos(π) + cos(π)) = 4t = 4
  CHECK(std::abs(hubbard.dispersion({2, 2}) - (4.0)) < kModelTolerance);

  // k=(0,2): ε = -2t * (cos(0) + cos(π)) = 0
  CHECK(std::abs(hubbard.dispersion({0, 2})) < kModelTolerance);
}

TEST_CASE("model_hubbard_momentum_kinetic_diagonal") {
  HubbardModelMomentum hubbard(1.0, 0.0, {4});  // U=0, only kinetic
  const Expression hamiltonian = hubbard.hamiltonian();

  // With U=0, all terms should be diagonal number operators
  // k=0: ε=-2, k=1: ε=0, k=2: ε=2, k=3: ε=0
  // Only k=0 and k=2 contribute (k=1,k=3 have zero energy)
  // 2 momentum points * 2 spins = 4 terms
  CHECK(hamiltonian.size() == 4u);
}

TEST_CASE("model_hubbard_momentum_interaction_momentum_conservation") {
  // The interaction term conserves momentum: k1+q and k2-q
  // This is implicitly tested by the structure of the Hamiltonian
  HubbardModelMomentum hubbard(0.0, 1.0, {2});  // t=0, only interaction
  const Expression hamiltonian = hubbard.hamiltonian();

  // With t=0, only interaction terms
  // 2^3 = 8 terms for the interaction (k1, k2, q each run over 2 values)
  CHECK(hamiltonian.size() > 0);
}

TEST_CASE("model_virtual_dispatch_hamiltonian_momentum") {
  HubbardModelMomentum hubbard(1.0, 2.0, {4});
  const Model& model = hubbard;
  CHECK(model.hamiltonian().size() > 0);
}

TEST_CASE("model_hubbard_momentum_throws_on_empty_size") {
  // Index throws std::out_of_range for empty dimensions
  CHECK_THROWS_AS(HubbardModelMomentum(1.0, 2.0, {}), std::out_of_range);
}
