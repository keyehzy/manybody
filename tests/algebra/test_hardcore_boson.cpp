#include <catch2/catch.hpp>

#include "algebra/hardcore_boson/basis.h"
#include "algebra/hardcore_boson/expression.h"
#include "algebra/matrix_elements.h"

using Op = HardcoreBosonOperator;
using Spin = Op::Spin;

// --- Canonicalization / normal ordering ---

TEST_CASE("hardcore_boson: single operator is identity") {
  Op op = Op::creation(Spin::Up, 0);
  HardcoreBosonMonomial term(op);
  HardcoreBosonExpression result = canonicalize(term);

  HardcoreBosonExpression::container_type ops{op};
  auto it = result.terms().find(ops);
  REQUIRE(it != result.terms().end());
  CHECK(it->second == HardcoreBosonExpression::complex_type(1.0, 0.0));
}

TEST_CASE("hardcore_boson: pauli exclusion — (a†)^2 = 0") {
  Op a = Op::creation(Spin::Up, 0);
  HardcoreBosonMonomial term({a, a});
  HardcoreBosonExpression result = canonicalize(term);
  CHECK(result.empty());
}

TEST_CASE("hardcore_boson: pauli exclusion — (a)^2 = 0") {
  Op a = Op::annihilation(Spin::Up, 0);
  HardcoreBosonMonomial term({a, a});
  HardcoreBosonExpression result = canonicalize(term);
  CHECK(result.empty());
}

TEST_CASE("hardcore_boson: on-site anticommutator {a, a†} = 1") {
  // a_0 a†_0 should canonicalize to 1 - a†_0 a_0
  Op create = Op::creation(Spin::Up, 0);
  Op annihilate = Op::annihilation(Spin::Up, 0);
  HardcoreBosonMonomial term({annihilate, create});
  HardcoreBosonExpression result = canonicalize(term);

  // Should have two terms: identity (coeff 1) and a†a (coeff -1)
  CHECK(result.size() == 2);

  HardcoreBosonExpression::container_type empty{};
  auto it_identity = result.terms().find(empty);
  REQUIRE(it_identity != result.terms().end());
  CHECK(it_identity->second == HardcoreBosonExpression::complex_type(1.0, 0.0));

  HardcoreBosonExpression::container_type normal{create, annihilate};
  auto it_normal = result.terms().find(normal);
  REQUIRE(it_normal != result.terms().end());
  CHECK(it_normal->second == HardcoreBosonExpression::complex_type(-1.0, 0.0));
}

TEST_CASE("hardcore_boson: different-site operators commute (no sign)") {
  // a†_1 a†_0 should canonicalize to +a†_0 a†_1 (swap_sign = +1)
  Op a0 = Op::creation(Spin::Up, 0);
  Op a1 = Op::creation(Spin::Up, 1);
  HardcoreBosonMonomial term({a1, a0});
  HardcoreBosonExpression result = canonicalize(term);

  CHECK(result.size() == 1);
  HardcoreBosonExpression::container_type ordered{a0, a1};
  auto it = result.terms().find(ordered);
  REQUIRE(it != result.terms().end());
  // No sign change for hardcore bosons (swap_sign = +1)
  CHECK(it->second == HardcoreBosonExpression::complex_type(1.0, 0.0));
}

TEST_CASE("hardcore_boson: different-site creation/annihilation commute") {
  // a_1 a†_0 should just swap to a†_0 a_1 with no contraction (different sites commute)
  Op create0 = Op::creation(Spin::Up, 0);
  Op annihilate1 = Op::annihilation(Spin::Up, 1);
  HardcoreBosonMonomial term({annihilate1, create0});
  HardcoreBosonExpression result = canonicalize(term);

  CHECK(result.size() == 1);
  HardcoreBosonExpression::container_type ordered{create0, annihilate1};
  auto it = result.terms().find(ordered);
  REQUIRE(it != result.terms().end());
  CHECK(it->second == HardcoreBosonExpression::complex_type(1.0, 0.0));
}

// --- Number operator properties ---

TEST_CASE("hardcore_boson: n^2 = n (projector property)") {
  // n_i^2 = a†_i a_i a†_i a_i = a†_i (1 - a†_i a_i) a_i = a†_i a_i = n_i
  auto n = HardcoreBosonExpression(hardcore_boson::number_op(Spin::Up, 0));
  HardcoreBosonExpression n_squared = canonicalize(n * n);
  HardcoreBosonExpression n_canon = canonicalize(n);

  CHECK(n_squared.size() == n_canon.size());
  for (const auto& [ops, coeff] : n_canon.terms()) {
    auto it = n_squared.terms().find(ops);
    REQUIRE(it != n_squared.terms().end());
    CHECK(std::abs(it->second - coeff) < 1e-12);
  }
}

// --- Commutator identities ---

TEST_CASE("hardcore_boson: [n_i, a†_i] = a†_i") {
  auto n = HardcoreBosonExpression(hardcore_boson::number_op(Spin::Up, 0));
  auto adag = HardcoreBosonExpression(hardcore_boson::creation(Spin::Up, 0));
  HardcoreBosonExpression comm = canonicalize(commutator(n, adag));

  CHECK(comm.size() == 1);
  Op create = Op::creation(Spin::Up, 0);
  HardcoreBosonExpression::container_type ops{create};
  auto it = comm.terms().find(ops);
  REQUIRE(it != comm.terms().end());
  CHECK(std::abs(it->second - HardcoreBosonExpression::complex_type(1.0, 0.0)) < 1e-12);
}

TEST_CASE("hardcore_boson: [n_i, a_i] = -a_i") {
  auto n = HardcoreBosonExpression(hardcore_boson::number_op(Spin::Up, 0));
  auto a = HardcoreBosonExpression(hardcore_boson::annihilation(Spin::Up, 0));
  HardcoreBosonExpression comm = canonicalize(commutator(n, a));

  CHECK(comm.size() == 1);
  Op annihilate = Op::annihilation(Spin::Up, 0);
  HardcoreBosonExpression::container_type ops{annihilate};
  auto it = comm.terms().find(ops);
  REQUIRE(it != comm.terms().end());
  CHECK(std::abs(it->second - HardcoreBosonExpression::complex_type(-1.0, 0.0)) < 1e-12);
}

// --- Basis generation ---

TEST_CASE("hardcore_boson: basis size matches fermion (exclusion)") {
  // 3 orbitals, 2 particles, spinless (both Up) = C(3,2) = 3 states
  auto basis = HardcoreBosonBasis::with_fixed_particle_number_and_spin(3, 2, 2);
  CHECK(basis.set.size() == 3);
}

TEST_CASE("hardcore_boson: basis with spin") {
  // 2 orbitals, 2 particles (1 up, 1 down) = C(2,1)*C(2,1) = 4 states
  auto basis = HardcoreBosonBasis::with_fixed_particle_number_and_spin(2, 2, 0);
  CHECK(basis.set.size() == 4);
}

TEST_CASE("hardcore_boson: state normalization is always 1") {
  auto basis = HardcoreBosonBasis::with_fixed_particle_number_and_spin(3, 2, 2);
  for (size_t i = 0; i < basis.set.size(); ++i) {
    CHECK(basis.state_normalization(i) == 1.0);
  }
}
