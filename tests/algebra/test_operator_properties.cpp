#include <rapidcheck.h>
#include <rapidcheck/catch.h>

#include "algebra/operator.h"

namespace {
constexpr std::size_t kValueLimit = Operator::max_index() + 1;
}

namespace rc {

template <>
struct Arbitrary<Operator::Type> {
  static Gen<Operator::Type> arbitrary() {
    return gen::element(Operator::Type::Creation, Operator::Type::Annihilation);
  }
};

template <>
struct Arbitrary<Operator::Spin> {
  static Gen<Operator::Spin> arbitrary() {
    return gen::element(Operator::Spin::Up, Operator::Spin::Down);
  }
};

template <>
struct Arbitrary<Operator> {
  static Gen<Operator> arbitrary() {
    using RawType = Operator::storage_type;
    return gen::map(gen::arbitrary<RawType>(), [](RawType x) { return Operator(x); });
  }
};

}  // namespace rc

TEST_CASE("Operator property tests") {
  using Type = Operator::Type;
  using Spin = Operator::Spin;

  // Property 1: Round-trip construction
  rc::prop("round-trip construction preserves type, spin, and value", [](Type type, Spin spin) {
    const auto value = *rc::gen::inRange<std::size_t>(0, kValueLimit);
    Operator op(type, spin, value);
    RC_ASSERT(op.type() == type);
    RC_ASSERT(op.spin() == spin);
    RC_ASSERT(op.value() == value);
  });

  // Property 2: Adjoint involution
  rc::prop("adjoint is an involution",
           [](Operator op) { RC_ASSERT(op.adjoint().adjoint() == op); });

  // Property 3: Adjoint preservation
  rc::prop("adjoint only changes type, preserves spin and value", [](Operator op) {
    RC_ASSERT(op.adjoint().type() != op.type());
    RC_ASSERT(op.adjoint().spin() == op.spin());
    RC_ASSERT(op.adjoint().value() == op.value());
  });

  // Property 4: Flip involution
  rc::prop("flip is an involution", [](Operator op) { RC_ASSERT(op.flip().flip() == op); });

  // Property 5: Flip preservation
  rc::prop("flip only changes spin, preserves type and value", [](Operator op) {
    RC_ASSERT(op.flip().spin() != op.spin());
    RC_ASSERT(op.flip().type() == op.type());
    RC_ASSERT(op.flip().value() == op.value());
  });

  // Property 6: Adjoint-flip commutativity
  rc::prop("adjoint and flip commute",
           [](Operator op) { RC_ASSERT(op.adjoint().flip() == op.flip().adjoint()); });

  // Property 7: Commutation symmetry
  rc::prop("commutes relation is symmetric",
           [](Operator a, Operator b) { RC_ASSERT(a.commutes(b) == b.commutes(a)); });

  // Property 8: Self-commutation
  rc::prop("every operator commutes with itself", [](Operator op) { RC_ASSERT(op.commutes(op)); });

  // Property 9: Adjoint non-commutation
  rc::prop("an operator does not commute with its adjoint",
           [](Operator op) { RC_ASSERT(!op.commutes(op.adjoint())); });

  // Property 10: Ordering irreflexivity
  rc::prop("no operator is less than itself", [](Operator op) { RC_ASSERT(!(op < op)); });

  // Property 11: Ordering asymmetry
  rc::prop("if a < b then not b < a", [](Operator a, Operator b) {
    if (a < b) {
      RC_ASSERT(!(b < a));
    }
  });

  // Property 12: Ordering transitivity
  rc::prop("ordering is transitive", [](Operator a, Operator b, Operator c) {
    if ((a < b) && (b < c)) {
      RC_ASSERT(a < c);
    }
  });

  // Property 13: Ordering trichotomy
  rc::prop("exactly one of a < b, a == b, or b < a holds", [](Operator a, Operator b) {
    int count = (a < b ? 1 : 0) + (a == b ? 1 : 0) + (b < a ? 1 : 0);
    RC_ASSERT(count == 1);
  });

  // Property 14: Equality reflexivity
  rc::prop("every operator equals itself", [](Operator op) { RC_ASSERT(op == op); });

  // Property 15: Equality symmetry
  rc::prop("equality is symmetric",
           [](Operator a, Operator b) { RC_ASSERT((a == b) == (b == a)); });

  // Property 16: Equality consistency with data
  rc::prop("operators are equal iff their data bytes are equal",
           [](Operator a, Operator b) { RC_ASSERT((a == b) == (a.data == b.data)); });

  // Property 17: Hash consistency
  rc::prop("equal operators have equal hashes", [](Operator a, Operator b) {
    if (a == b) {
      std::hash<Operator> hasher;
      RC_ASSERT(hasher(a) == hasher(b));
    }
  });

  // Property 18: Factory method equivalence
  rc::prop("factory methods produce same result as constructor", [](Spin spin) {
    const auto value = *rc::gen::inRange<std::size_t>(0, kValueLimit);
    RC_ASSERT(Operator::creation(spin, value) == Operator(Type::Creation, spin, value));
    RC_ASSERT(Operator::annihilation(spin, value) == Operator(Type::Annihilation, spin, value));
  });
}
