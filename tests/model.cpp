#include "models/model.h"

#include <type_traits>

#include "framework.h"
#include "models/hubbard_model.h"

TEST(model_hubbard_inherits_from_interface) {
  EXPECT_TRUE((std::is_base_of_v<Model, HubbardModel>));
}

TEST(model_hubbard_2d_inherits_from_interface) {
  EXPECT_TRUE((std::is_base_of_v<Model, HubbardModel2D>));
}

TEST(model_hubbard_3d_inherits_from_interface) {
  EXPECT_TRUE((std::is_base_of_v<Model, HubbardModel3D>));
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
