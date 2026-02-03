#include <catch2/catch.hpp>

#include "algebra/majorana/majorana_commutator.h"

namespace majorana_commutator_tests {

static MajoranaString make_string(std::initializer_list<MajoranaElement> elements) {
  MajoranaString str;
  str.data.append_range(elements.begin(), elements.end());
  return str;
}

static MajoranaElement even(size_t orbital, Operator::Spin spin) {
  return MajoranaElement::even(orbital, spin);
}

static MajoranaElement odd(size_t orbital, Operator::Spin spin) {
  return MajoranaElement::odd(orbital, spin);
}

TEST_CASE("majorana_clifford_anticommutator_same_index") {
  // {gamma_i, gamma_i} = 2
  MajoranaString str = make_string({odd(0, Operator::Spin::Down)});
  MajoranaExpression gi(MajoranaExpression::complex_type(1.0, 0.0), str);

  auto result = anticommutator(gi, gi);

  MajoranaString empty;
  CHECK(result.size() == 1u);
  auto it = result.hashmap.find(empty);
  CHECK(it != result.hashmap.end());
  CHECK(std::abs(it->second - MajoranaExpression::complex_type(2.0, 0.0)) < 1e-12);
}

TEST_CASE("majorana_clifford_anticommutator_different_indices") {
  // {gamma_i, gamma_j} = 0 for i != j
  MajoranaString str_i = make_string({odd(0, Operator::Spin::Up)});
  MajoranaString str_j = make_string({even(1, Operator::Spin::Down)});
  MajoranaExpression gi(MajoranaExpression::complex_type(1.0, 0.0), str_i);
  MajoranaExpression gj(MajoranaExpression::complex_type(1.0, 0.0), str_j);

  auto result = anticommutator(gi, gj);

  CHECK(result.size() == 0u);
}

TEST_CASE("majorana_commutator_same_index_vanishes") {
  // [gamma_i, gamma_i] = 0
  MajoranaString str = make_string({odd(1, Operator::Spin::Down)});
  MajoranaExpression gi(MajoranaExpression::complex_type(1.0, 0.0), str);

  auto result = commutator(gi, gi);

  CHECK(result.size() == 0u);
}

TEST_CASE("majorana_commutator_different_indices") {
  // [gamma_i, gamma_j] = 2 * gamma_i * gamma_j for i != j
  MajoranaString str_i = make_string({even(0, Operator::Spin::Down)});
  MajoranaString str_j = make_string({even(1, Operator::Spin::Up)});
  MajoranaExpression gi(MajoranaExpression::complex_type(1.0, 0.0), str_i);
  MajoranaExpression gj(MajoranaExpression::complex_type(1.0, 0.0), str_j);

  auto result = commutator(gi, gj);

  MajoranaString expected =
      make_string({even(0, Operator::Spin::Down), even(1, Operator::Spin::Up)});
  CHECK(result.size() == 1u);
  auto it = result.hashmap.find(expected);
  CHECK(it != result.hashmap.end());
  CHECK(std::abs(it->second - MajoranaExpression::complex_type(2.0, 0.0)) < 1e-12);
}

TEST_CASE("majorana_commutator_distributes_over_sum") {
  MajoranaString str_a = make_string({even(0, Operator::Spin::Up)});
  MajoranaString str_b = make_string({even(0, Operator::Spin::Down)});
  MajoranaString str_c = make_string({odd(0, Operator::Spin::Up)});
  MajoranaExpression a(MajoranaExpression::complex_type(1.0, 0.0), str_a);
  MajoranaExpression b(MajoranaExpression::complex_type(1.0, 0.0), str_b);
  MajoranaExpression c(MajoranaExpression::complex_type(1.0, 0.0), str_c);

  MajoranaExpression sum = a + b;

  auto result = commutator(sum, c);
  auto expected = commutator(a, c) + commutator(b, c);

  CHECK(result.size() == expected.size());
  for (const auto& [str, coeff] : expected.hashmap) {
    auto it = result.hashmap.find(str);
    CHECK(it != result.hashmap.end());
    CHECK(std::abs(it->second - coeff) < 1e-12);
  }
}

TEST_CASE("majorana_anticommutator_distributes_over_sum") {
  MajoranaString str_a = make_string({even(0, Operator::Spin::Up)});
  MajoranaString str_b = make_string({even(0, Operator::Spin::Down)});
  MajoranaString str_c = make_string({odd(0, Operator::Spin::Up)});
  MajoranaExpression a(MajoranaExpression::complex_type(1.0, 0.0), str_a);
  MajoranaExpression b(MajoranaExpression::complex_type(1.0, 0.0), str_b);
  MajoranaExpression c(MajoranaExpression::complex_type(1.0, 0.0), str_c);

  MajoranaExpression sum = a + b;

  auto result = anticommutator(sum, c);
  auto expected = anticommutator(a, c) + anticommutator(b, c);

  CHECK(result.size() == expected.size());
  for (const auto& [str, coeff] : expected.hashmap) {
    auto it = result.hashmap.find(str);
    CHECK(it != result.hashmap.end());
    CHECK(std::abs(it->second - coeff) < 1e-12);
  }
}

TEST_CASE("majorana_commutator_bilinear_identity") {
  // [gamma_i gamma_j, gamma_k] with i < j, k different from both
  // should be non-zero
  MajoranaString str_ij = make_string({even(0, Operator::Spin::Up), even(0, Operator::Spin::Down)});
  MajoranaString str_k = make_string({odd(0, Operator::Spin::Up)});
  MajoranaExpression ij(MajoranaExpression::complex_type(1.0, 0.0), str_ij);
  MajoranaExpression gk(MajoranaExpression::complex_type(1.0, 0.0), str_k);

  auto result = commutator(ij, gk);

  // [gamma_0 gamma_1, gamma_2]:
  //   AB = gamma_0 gamma_1 gamma_2, sign from multiply_strings({0,1}, {2}) = +1
  //   BA = gamma_2 gamma_0 gamma_1, sign from multiply_strings({2}, {0,1}) = +1
  //   Both signs are +1, so commutator vanishes
  // Actually: multiply_strings({0,1}, {2}): b[0]=2 passes 0 remaining a => sign = +1
  //           multiply_strings({2}, {0,1}): b[0]=0 passes 1 remaining a(2), b[1]=1 passes 1
  //           remaining a(2) => total 2 swaps => sign = +1
  // Same signs => commutator is 0 for this case
  CHECK(result.size() == 0u);
}

TEST_CASE("majorana_commutator_string_with_overlap") {
  // [gamma_0 gamma_1, gamma_0 gamma_2]
  MajoranaString str_a = make_string({even(0, Operator::Spin::Up), even(0, Operator::Spin::Down)});
  MajoranaString str_b = make_string({even(0, Operator::Spin::Up), odd(0, Operator::Spin::Up)});
  MajoranaExpression a(MajoranaExpression::complex_type(1.0, 0.0), str_a);
  MajoranaExpression b(MajoranaExpression::complex_type(1.0, 0.0), str_b);

  auto result = commutator(a, b);

  // AB = (gamma_0 gamma_1)(gamma_0 gamma_2)
  //   multiply_strings({0,1}, {0,2}):
  //   a[0]=0 == b[0]=0 => cancel, passes na-i-1 = 1 remaining a => sign *= -1 => sign = -1
  //   a[1]=1 < b[1]=2 => push 1
  //   remaining b: push 2
  //   result: {1,2} with sign -1
  //
  // BA = (gamma_0 gamma_2)(gamma_0 gamma_1)
  //   multiply_strings({0,2}, {0,1}):
  //   a[0]=0 == b[0]=0 => cancel, passes na-i-1 = 1 remaining a => sign *= -1 => sign = -1
  //   a[1]=2 > b[1]=1 => b[1]=1 passes 1 remaining a => sign *= -1 => sign = +1, push 1
  //   remaining a: push 2
  //   result: {1,2} with sign +1
  //
  // Signs differ => commutator = 2 * (-1) * {1,2} = -2 * gamma_1 gamma_2
  MajoranaString expected =
      make_string({even(0, Operator::Spin::Down), odd(0, Operator::Spin::Up)});
  CHECK(result.size() == 1u);
  auto it = result.hashmap.find(expected);
  CHECK(it != result.hashmap.end());
  CHECK(std::abs(it->second - MajoranaExpression::complex_type(-2.0, 0.0)) < 1e-12);
}

}  // namespace majorana_commutator_tests
