#include "models/model.h"

#include <type_traits>

#include "framework.h"
#include "models/hubbard_model.h"

TEST(model_hubbard_inherits_from_interface) {
  EXPECT_TRUE((std::is_base_of_v<Model, HubbardModel>));
}

TEST(model_hubbard_hamiltonian_term_count) {
  HubbardModel hubbard(1.0, 2.0, 2);
  const Expression hamiltonian = hubbard.hamiltonian();
  EXPECT_EQ(hamiltonian.size(), 6u);
}

TEST(model_virtual_dispatch_hamiltonian) {
  HubbardModel hubbard(1.0, 2.0, 2);
  const Model& model = hubbard;
  EXPECT_EQ(model.hamiltonian().size(), 6u);
}
