#include "models/model.h"

#include <cmath>
#include <numbers>
#include <type_traits>

#include "framework.h"
#include "models/hubbard_model.h"
#include "models/hubbard_model_relative.h"

namespace {
constexpr double kModelTolerance = 1e-6;

bool complex_near_model(const Term::complex_type& lhs, const Term::complex_type& rhs, double tol) {
  const auto delta = lhs - rhs;
  return std::abs(delta) <= tol;
}
}  // namespace

TEST(model_hubbard_inherits_from_interface) {
  EXPECT_TRUE((std::is_base_of_v<Model, HubbardModel>));
}

TEST(model_hubbard_2d_inherits_from_interface) {
  EXPECT_TRUE((std::is_base_of_v<Model, HubbardModel2D>));
}

TEST(model_hubbard_3d_inherits_from_interface) {
  EXPECT_TRUE((std::is_base_of_v<Model, HubbardModel3D>));
}

TEST(model_hubbard_relative_inherits_from_interface) {
  EXPECT_TRUE((std::is_base_of_v<Model, HubbardModelRelative>));
}

TEST(model_hubbard_hamiltonian_term_count) {
  HubbardModel hubbard(1.0, 2.0, 2);
  const Expression hamiltonian = hubbard.hamiltonian();
  EXPECT_EQ(hamiltonian.size(), 6u);
}

TEST(model_hubbard_2d_hamiltonian_term_count) {
  HubbardModel2D hubbard(1.0, 2.0, 2, 2);
  const Expression hamiltonian = hubbard.hamiltonian();
  EXPECT_EQ(hamiltonian.size(), 20u);
}

TEST(model_hubbard_3d_hamiltonian_term_count) {
  HubbardModel3D hubbard(1.0, 2.0, 2, 2, 2);
  const Expression hamiltonian = hubbard.hamiltonian();
  EXPECT_EQ(hamiltonian.size(), 56u);
}

TEST(model_hubbard_relative_hamiltonian_term_count) {
  HubbardModelRelative hubbard(1.0, 2.0, 4, 0);
  const Expression hamiltonian = hubbard.hamiltonian();
  EXPECT_EQ(hamiltonian.size(), 16u);
}

TEST(model_hubbard_relative_hamiltonian_coefficients) {
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
  EXPECT_TRUE(diag_it != hamiltonian.hashmap.end());
  EXPECT_TRUE(off_diag_it != hamiltonian.hashmap.end());

  const double k_phase =
      2.0 * std::numbers::pi_v<double> * static_cast<double>(p_same) / static_cast<double>(size);
  const double k_total_phase =
      2.0 * std::numbers::pi_v<double> * static_cast<double>(kMomentum) / static_cast<double>(size);
  const double t_eff = 2.0 * 1.25 * std::cos(0.5 * k_total_phase);
  const double expected_diag = (3.0 / static_cast<double>(size)) + 2.0 * t_eff * std::cos(k_phase);
  const double expected_off_diag = 3.0 / static_cast<double>(size);

  EXPECT_TRUE(complex_near_model(diag_it->second,
                                 Term::complex_type(static_cast<float>(expected_diag), 0.0f),
                                 kModelTolerance));
  EXPECT_TRUE(complex_near_model(off_diag_it->second,
                                 Term::complex_type(static_cast<float>(expected_off_diag), 0.0f),
                                 kModelTolerance));
}

TEST(model_virtual_dispatch_hamiltonian) {
  HubbardModel hubbard(1.0, 2.0, 2);
  const Model& model = hubbard;
  EXPECT_EQ(model.hamiltonian().size(), 6u);
}

TEST(model_virtual_dispatch_hamiltonian_2d) {
  HubbardModel2D hubbard(1.0, 2.0, 2, 2);
  const Model& model = hubbard;
  EXPECT_EQ(model.hamiltonian().size(), 20u);
}

TEST(model_virtual_dispatch_hamiltonian_3d) {
  HubbardModel3D hubbard(1.0, 2.0, 2, 2, 2);
  const Model& model = hubbard;
  EXPECT_EQ(model.hamiltonian().size(), 56u);
}

TEST(model_virtual_dispatch_hamiltonian_relative) {
  HubbardModelRelative hubbard(1.0, 2.0, 4, 0);
  const Model& model = hubbard;
  EXPECT_EQ(model.hamiltonian().size(), 16u);
}
