#include <catch2/catch.hpp>
#include <cmath>
#include <type_traits>

#include "algebra/boson/model/sawtooth_model.h"

namespace {

constexpr double kTolerance = 1e-12;

bool complex_near(const BosonMonomial::complex_type& lhs, const BosonMonomial::complex_type& rhs) {
  return std::abs(lhs - rhs) < kTolerance;
}

}  // namespace

TEST_CASE("sawtooth_model_inherits_from_boson_model_interface") {
  CHECK((std::is_base_of_v<BasicModel<BosonExpression>, SawtoothHubbardModel>));
}

TEST_CASE("sawtooth_model_basic_properties") {
  SawtoothHubbardModel model(1.0, std::sqrt(2.0), 4.0, 5);

  CHECK(model.t_base == 1.0);
  CHECK(model.t_tooth == Approx(std::sqrt(2.0)));
  CHECK(model.u == 4.0);
  CHECK(model.num_cells == 5);
  CHECK(model.num_sites == 10);

  for (size_t cell = 0; cell < model.num_cells; ++cell) {
    CHECK(model.site_base(cell) == 2 * cell);
    CHECK(model.site_apex(cell) == 2 * cell + 1);
    CHECK(model.site(SawtoothHubbardModel::SUBLATTICE_BASE, cell) == 2 * cell);
    CHECK(model.site(SawtoothHubbardModel::SUBLATTICE_APEX, cell) == 2 * cell + 1);
  }

  CHECK(model.site_base(0, -1) == model.site_base(model.num_cells - 1));
  CHECK(model.site_base(model.num_cells - 1, 1) == model.site_base(0));
  CHECK(model.site_apex(0, -1) == model.site_apex(model.num_cells - 1));
  CHECK(model.site_apex(model.num_cells - 1, 1) == model.site_apex(0));
}

TEST_CASE("sawtooth_model_term_counts_match_geometry") {
  SawtoothHubbardModel model(1.0, 1.5, 3.0, 3);

  const BosonExpression kinetic = model.kinetic();
  const BosonExpression interaction = model.interaction();
  const BosonExpression hamiltonian = model.hamiltonian();

  CHECK(kinetic.size() == 18u);
  CHECK(interaction.size() == 6u);
  CHECK(hamiltonian.size() == 24u);
}

TEST_CASE("sawtooth_model_hamiltonian_contains_expected_terms") {
  SawtoothHubbardModel model(1.25, 2.0, 6.0, 4);
  const BosonExpression hamiltonian = model.hamiltonian();

  BosonExpression::container_type base_forward{
      BosonOperator::creation(SawtoothHubbardModel::species, model.site_base(0)),
      BosonOperator::annihilation(SawtoothHubbardModel::species, model.site_base(1)),
  };
  BosonExpression::container_type base_backward{
      BosonOperator::creation(SawtoothHubbardModel::species, model.site_base(1)),
      BosonOperator::annihilation(SawtoothHubbardModel::species, model.site_base(0)),
  };
  BosonExpression::container_type left_tooth{
      BosonOperator::creation(SawtoothHubbardModel::species, model.site_base(0)),
      BosonOperator::annihilation(SawtoothHubbardModel::species, model.site_apex(0)),
  };
  BosonExpression::container_type right_tooth{
      BosonOperator::creation(SawtoothHubbardModel::species, model.site_base(1)),
      BosonOperator::annihilation(SawtoothHubbardModel::species, model.site_apex(0)),
  };
  BosonExpression::container_type onsite_base{
      BosonOperator::creation(SawtoothHubbardModel::species, model.site_base(0)),
      BosonOperator::creation(SawtoothHubbardModel::species, model.site_base(0)),
      BosonOperator::annihilation(SawtoothHubbardModel::species, model.site_base(0)),
      BosonOperator::annihilation(SawtoothHubbardModel::species, model.site_base(0)),
  };
  BosonExpression::container_type onsite_apex{
      BosonOperator::creation(SawtoothHubbardModel::species, model.site_apex(0)),
      BosonOperator::creation(SawtoothHubbardModel::species, model.site_apex(0)),
      BosonOperator::annihilation(SawtoothHubbardModel::species, model.site_apex(0)),
      BosonOperator::annihilation(SawtoothHubbardModel::species, model.site_apex(0)),
  };

  const auto base_forward_it = hamiltonian.terms().find(base_forward);
  const auto base_backward_it = hamiltonian.terms().find(base_backward);
  const auto left_tooth_it = hamiltonian.terms().find(left_tooth);
  const auto right_tooth_it = hamiltonian.terms().find(right_tooth);
  const auto onsite_base_it = hamiltonian.terms().find(onsite_base);
  const auto onsite_apex_it = hamiltonian.terms().find(onsite_apex);

  REQUIRE(base_forward_it != hamiltonian.terms().end());
  REQUIRE(base_backward_it != hamiltonian.terms().end());
  REQUIRE(left_tooth_it != hamiltonian.terms().end());
  REQUIRE(right_tooth_it != hamiltonian.terms().end());
  REQUIRE(onsite_base_it != hamiltonian.terms().end());
  REQUIRE(onsite_apex_it != hamiltonian.terms().end());

  CHECK(complex_near(base_forward_it->second, BosonMonomial::complex_type(-1.25, 0.0)));
  CHECK(complex_near(base_backward_it->second, BosonMonomial::complex_type(-1.25, 0.0)));
  CHECK(complex_near(left_tooth_it->second, BosonMonomial::complex_type(-2.0, 0.0)));
  CHECK(complex_near(right_tooth_it->second, BosonMonomial::complex_type(-2.0, 0.0)));
  CHECK(complex_near(onsite_base_it->second, BosonMonomial::complex_type(3.0, 0.0)));
  CHECK(complex_near(onsite_apex_it->second, BosonMonomial::complex_type(3.0, 0.0)));
}
