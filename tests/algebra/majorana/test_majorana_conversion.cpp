#include <catch2/catch.hpp>

#include "algebra/majorana/conversion.h"
#include "utils/tolerances.h"

using namespace majorana;

static constexpr auto tol = tolerances::tolerance<double>();

namespace majorana_conversion_tests {

static MajoranaMonomial::container_type make_string(
    std::initializer_list<MajoranaOperator> elements) {
  MajoranaMonomial::container_type str;
  str.append_range(elements.begin(), elements.end());
  return str;
}

static MajoranaOperator even(size_t orbital, FermionOperator::Spin spin) {
  return MajoranaOperator::even(orbital, spin);
}

static MajoranaOperator odd(size_t orbital, FermionOperator::Spin spin) {
  return MajoranaOperator::odd(orbital, spin);
}

/// Helper: compare two Expressions term-by-term within tolerance.
static bool expressions_equal(const FermionExpression& a, const FermionExpression& b) {
  if (a.size() != b.size()) return false;
  for (const auto& [ops, coeff] : a.terms()) {
    auto it = b.terms().find(ops);
    if (it == b.terms().end()) return false;
    if (std::abs(coeff - it->second) > tol) return false;
  }
  return true;
}

TEST_CASE("majorana_conversion_single_creation") {
  // c+(Up, 0) = (gamma_0 + i * gamma_2) / 2
  auto op = FermionOperator::creation(FermionOperator::Spin::Up, 0);
  FermionExpression expr(op);

  auto maj = to_majorana(expr);

  MajoranaMonomial::container_type even_str = make_string({even(0, FermionOperator::Spin::Up)});
  MajoranaMonomial::container_type odd_str = make_string({odd(0, FermionOperator::Spin::Up)});
  CHECK(maj.size() == 2u);

  auto it_e = maj.terms().find(even_str);
  CHECK(it_e != maj.terms().end());
  CHECK(std::abs(it_e->second - MajoranaExpression::complex_type(0.5, 0.0)) < 1e-12);

  auto it_o = maj.terms().find(odd_str);
  CHECK(it_o != maj.terms().end());
  CHECK(std::abs(it_o->second - MajoranaExpression::complex_type(0.0, 0.5)) < 1e-12);
}

TEST_CASE("majorana_conversion_single_annihilation") {
  // c(Up, 0) = (gamma_0 - i * gamma_2) / 2
  auto op = FermionOperator::annihilation(FermionOperator::Spin::Up, 0);
  FermionExpression expr(op);

  auto maj = to_majorana(expr);

  MajoranaMonomial::container_type even_str = make_string({even(0, FermionOperator::Spin::Up)});
  MajoranaMonomial::container_type odd_str = make_string({odd(0, FermionOperator::Spin::Up)});
  CHECK(maj.size() == 2u);

  auto it_e = maj.terms().find(even_str);
  CHECK(it_e != maj.terms().end());
  CHECK(std::abs(it_e->second - MajoranaExpression::complex_type(0.5, 0.0)) < 1e-12);

  auto it_o = maj.terms().find(odd_str);
  CHECK(it_o != maj.terms().end());
  CHECK(std::abs(it_o->second - MajoranaExpression::complex_type(0.0, -0.5)) < 1e-12);
}

TEST_CASE("majorana_conversion_density_operator") {
  // n = c+(Up,0) c(Up,0)
  // In Majorana: (1/2)(1 + i gamma_0 gamma_2) using gamma_e=0, gamma_o=2
  auto n = density(FermionOperator::Spin::Up, 0);
  FermionExpression expr(n);

  auto maj = to_majorana(expr);

  MajoranaMonomial::container_type empty;
  MajoranaMonomial::container_type pair =
      make_string({even(0, FermionOperator::Spin::Up), odd(0, FermionOperator::Spin::Up)});

  auto it_id = maj.terms().find(empty);
  CHECK(it_id != maj.terms().end());
  CHECK(std::abs(it_id->second - MajoranaExpression::complex_type(0.5, 0.0)) < 1e-12);

  auto it_pair = maj.terms().find(pair);
  CHECK(it_pair != maj.terms().end());
  CHECK(std::abs(it_pair->second - MajoranaExpression::complex_type(0.0, -0.5)) < 1e-12);

  CHECK(maj.size() == 2u);
}

TEST_CASE("majorana_conversion_hopping_term") {
  // t * (c+(Up,0) c(Up,1) + c+(Up,1) c(Up,0))
  FermionExpression hop = hopping(1.0, 0, 1, FermionOperator::Spin::Up);

  auto maj = to_majorana(hop);

  // Hopping between site 0 and 1, spin Up:
  // Site 0 Up: gamma_0 (even), gamma_2 (odd)
  // Site 1 Up: gamma_4 (even), gamma_6 (odd)
  // The result should contain bilinear Majorana terms
  CHECK(maj.size() > 0u);

  // Round-trip: convert back and normal-order, should match
  FermionExpression roundtrip = from_majorana(maj);
  FermionExpression original = canonicalize(hop);

  CHECK(expressions_equal(roundtrip, original));
}

TEST_CASE("majorana_conversion_roundtrip_single_operator") {
  auto op = FermionOperator::creation(FermionOperator::Spin::Down, 2);
  FermionExpression original(op);

  auto maj = to_majorana(original);
  FermionExpression roundtrip = from_majorana(maj);
  FermionExpression expected = canonicalize(original);

  CHECK(expressions_equal(roundtrip, expected));
}

TEST_CASE("majorana_conversion_roundtrip_density") {
  FermionExpression original(density(FermionOperator::Spin::Up, 1));

  auto maj = to_majorana(original);
  FermionExpression roundtrip = from_majorana(maj);
  FermionExpression expected = canonicalize(original);

  CHECK(expressions_equal(roundtrip, expected));
}

TEST_CASE("majorana_conversion_roundtrip_two_body") {
  FermionExpression original(
      density_density(FermionOperator::Spin::Up, 0, FermionOperator::Spin::Down, 0));

  auto maj = to_majorana(original);
  FermionExpression roundtrip = from_majorana(maj);
  FermionExpression expected = canonicalize(original);

  CHECK(expressions_equal(roundtrip, expected));
}

TEST_CASE("majorana_conversion_tv_hamiltonian") {
  // t-V model on 2 sites (spinless-like using Up spin only):
  // H = -t sum (c+_i c_j + h.c.) + V sum n_i n_j
  const double t = 1.0;
  const double V = 0.5;
  const size_t L = 2;

  FermionExpression H;
  for (size_t i = 0; i < L; ++i) {
    size_t j = (i + 1) % L;
    H += hopping(t, i, j, FermionOperator::Spin::Up);
    H += FermionExpression(
        FermionExpression::complex_type(V) *
        density_density(FermionOperator::Spin::Up, i, FermionOperator::Spin::Up, j));
  }

  FermionExpression H_normal = canonicalize(H);

  auto H_maj = to_majorana(H);
  FermionExpression H_roundtrip = from_majorana(H_maj);

  CHECK(expressions_equal(H_roundtrip, H_normal));
}

TEST_CASE("majorana_conversion_roundtrip_scalar_complex") {
  FermionExpression original(FermionExpression::complex_type(0.25, -0.5));

  auto maj = to_majorana(original);
  FermionExpression roundtrip = from_majorana(maj);
  FermionExpression expected = canonicalize(original);

  CHECK(expressions_equal(roundtrip, expected));
}

TEST_CASE("majorana_conversion_from_majorana_even_gamma") {
  MajoranaMonomial::container_type str = make_string({even(1, FermionOperator::Spin::Down)});
  MajoranaExpression maj(MajoranaExpression::complex_type(1.0, 0.0), str);

  FermionExpression result = from_majorana(maj);

  FermionExpression expected;
  expected += FermionExpression(FermionOperator::creation(FermionOperator::Spin::Down, 1));
  expected += FermionExpression(FermionOperator::annihilation(FermionOperator::Spin::Down, 1));
  expected = canonicalize(expected);

  CHECK(expressions_equal(result, expected));
}

TEST_CASE("majorana_conversion_from_majorana_odd_gamma_complex_coeff") {
  MajoranaMonomial::container_type str = make_string({odd(0, FermionOperator::Spin::Up)});
  MajoranaExpression maj(MajoranaExpression::complex_type(0.0, 2.0), str);

  FermionExpression result = from_majorana(maj);

  FermionExpression expected;
  FermionOperator create = FermionOperator::creation(FermionOperator::Spin::Up, 0);
  FermionOperator annihilate = FermionOperator::annihilation(FermionOperator::Spin::Up, 0);
  expected +=
      FermionExpression(FermionMonomial(FermionExpression::complex_type(2.0, 0.0), {create}));
  expected +=
      FermionExpression(FermionMonomial(FermionExpression::complex_type(-2.0, 0.0), {annihilate}));
  expected = canonicalize(expected);

  CHECK(expressions_equal(result, expected));
}

TEST_CASE("majorana_conversion_roundtrip_complex_one_body") {
  FermionOperator create = FermionOperator::creation(FermionOperator::Spin::Up, 0);
  FermionOperator annihilate = FermionOperator::annihilation(FermionOperator::Spin::Down, 1);
  FermionExpression original(
      FermionMonomial(FermionExpression::complex_type(0.0, 0.75), {create, annihilate}));

  auto maj = to_majorana(original);
  FermionExpression roundtrip = from_majorana(maj);
  FermionExpression expected = canonicalize(original);

  CHECK(expressions_equal(roundtrip, expected));
}

}  // namespace majorana_conversion_tests
