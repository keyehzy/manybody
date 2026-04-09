#include <rapidcheck.h>
#include <rapidcheck/catch.h>

#include "algebra/operator.h"

namespace {
constexpr std::size_t kValueLimit = FermionOperator::max_index() + 1;
}

namespace rc {

template <>
struct Arbitrary<FermionOperator::Type> {
  static Gen<FermionOperator::Type> arbitrary() {
    return gen::element(FermionOperator::Type::Creation, FermionOperator::Type::Annihilation);
  }
};

template <>
struct Arbitrary<FermionOperator::Spin> {
  static Gen<FermionOperator::Spin> arbitrary() {
    return gen::element(FermionOperator::Spin::Up, FermionOperator::Spin::Down);
  }
};

template <>
struct Arbitrary<FermionOperator> {
  static Gen<FermionOperator> arbitrary() {
    using RawType = FermionOperator::storage_type;
    return gen::map(gen::arbitrary<RawType>(), [](RawType x) { return FermionOperator(x); });
  }
};

}  // namespace rc

TEST_CASE("FermionOperator property tests") {
  using Type = FermionOperator::Type;
  using Spin = FermionOperator::Spin;

  // Property 1: Round-trip construction
  rc::prop("round-trip construction preserves type, spin, and value", [](Type type, Spin spin) {
    const auto value = *rc::gen::inRange<std::size_t>(0, kValueLimit);
    FermionOperator op(type, spin, value);
    RC_ASSERT(op.type() == type);
    RC_ASSERT(op.spin() == spin);
    RC_ASSERT(op.value() == value);
  });

  // Property 2: Adjoint involution
  rc::prop("adjoint is an involution",
           [](FermionOperator op) { RC_ASSERT(op.adjoint().adjoint() == op); });

  // Property 3: Adjoint preservation
  rc::prop("adjoint only changes type, preserves spin and value", [](FermionOperator op) {
    RC_ASSERT(op.adjoint().type() != op.type());
    RC_ASSERT(op.adjoint().spin() == op.spin());
    RC_ASSERT(op.adjoint().value() == op.value());
  });

  // Property 4: Flip involution
  rc::prop("flip is an involution", [](FermionOperator op) { RC_ASSERT(op.flip().flip() == op); });

  // Property 5: Flip preservation
  rc::prop("flip only changes spin, preserves type and value", [](FermionOperator op) {
    RC_ASSERT(op.flip().spin() != op.spin());
    RC_ASSERT(op.flip().type() == op.type());
    RC_ASSERT(op.flip().value() == op.value());
  });

  // Property 6: Adjoint-flip commutativity
  rc::prop("adjoint and flip commute",
           [](FermionOperator op) { RC_ASSERT(op.adjoint().flip() == op.flip().adjoint()); });

  // Property 7: Commutation symmetry
  rc::prop("commutes relation is symmetric",
           [](FermionOperator a, FermionOperator b) { RC_ASSERT(a.commutes(b) == b.commutes(a)); });

  // Property 8: Self-commutation
  rc::prop("every operator commutes with itself",
           [](FermionOperator op) { RC_ASSERT(op.commutes(op)); });

  // Property 9: Adjoint non-commutation
  rc::prop("an operator does not commute with its adjoint",
           [](FermionOperator op) { RC_ASSERT(!op.commutes(op.adjoint())); });

  // Property 10: Ordering irreflexivity
  rc::prop("no operator is less than itself", [](FermionOperator op) { RC_ASSERT(!(op < op)); });

  // Property 11: Ordering asymmetry
  rc::prop("if a < b then not b < a", [](FermionOperator a, FermionOperator b) {
    if (a < b) {
      RC_ASSERT(!(b < a));
    }
  });

  // Property 12: Ordering transitivity
  rc::prop("ordering is transitive", [](FermionOperator a, FermionOperator b, FermionOperator c) {
    if ((a < b) && (b < c)) {
      RC_ASSERT(a < c);
    }
  });

  // Property 13: Ordering trichotomy
  rc::prop("exactly one of a < b, a == b, or b < a holds",
           [](FermionOperator a, FermionOperator b) {
             int count = (a < b ? 1 : 0) + (a == b ? 1 : 0) + (b < a ? 1 : 0);
             RC_ASSERT(count == 1);
           });

  // Property 14: Equality reflexivity
  rc::prop("every operator equals itself", [](FermionOperator op) { RC_ASSERT(op == op); });

  // Property 15: Equality symmetry
  rc::prop("equality is symmetric",
           [](FermionOperator a, FermionOperator b) { RC_ASSERT((a == b) == (b == a)); });

  // Property 16: Equality consistency with data
  rc::prop("operators are equal iff their data bytes are equal",
           [](FermionOperator a, FermionOperator b) { RC_ASSERT((a == b) == (a.data == b.data)); });

  // Property 17: Hash consistency
  rc::prop("equal operators have equal hashes", [](FermionOperator a, FermionOperator b) {
    if (a == b) {
      std::hash<FermionOperator> hasher;
      RC_ASSERT(hasher(a) == hasher(b));
    }
  });

  // Property 18: Factory method equivalence
  rc::prop("factory methods produce same result as constructor", [](Spin spin) {
    const auto value = *rc::gen::inRange<std::size_t>(0, kValueLimit);
    RC_ASSERT(FermionOperator::creation(spin, value) ==
              FermionOperator(Type::Creation, spin, value));
    RC_ASSERT(FermionOperator::annihilation(spin, value) ==
              FermionOperator(Type::Annihilation, spin, value));
  });
}
