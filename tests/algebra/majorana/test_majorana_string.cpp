#include <catch2/catch.hpp>

#include "algebra/majorana/majorana_string.h"

namespace majorana_string_tests {

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

TEST_CASE("majorana_string_disjoint_multiply_concatenates") {
  MajoranaString a = make_string(
      {even(0, Operator::Spin::Up), odd(0, Operator::Spin::Up), even(1, Operator::Spin::Up)});
  MajoranaString b = make_string(
      {even(0, Operator::Spin::Down), odd(0, Operator::Spin::Down), even(1, Operator::Spin::Down)});

  auto result = multiply_strings(a, b);

  MajoranaString expected = make_string(
      {even(0, Operator::Spin::Up), even(0, Operator::Spin::Down), odd(0, Operator::Spin::Up),
       odd(0, Operator::Spin::Down), even(1, Operator::Spin::Up), even(1, Operator::Spin::Down)});
  CHECK(result.string == expected);
  // b[0]=1 passes 2 a-elements (2,4), b[1]=3 passes 1 (4), b[2]=5 passes 0 => 3 swaps (odd)
  CHECK(result.sign == -1);
}

TEST_CASE("majorana_string_disjoint_multiply_odd_sign") {
  MajoranaString a = make_string({even(0, Operator::Spin::Up), odd(0, Operator::Spin::Up)});
  MajoranaString b = make_string({even(0, Operator::Spin::Down)});

  auto result = multiply_strings(a, b);

  MajoranaString expected = make_string(
      {even(0, Operator::Spin::Up), even(0, Operator::Spin::Down), odd(0, Operator::Spin::Up)});
  CHECK(result.string == expected);
  // b[0]=1 passes 1 remaining a-element (a[1]=2) => 1 swap (odd)
  CHECK(result.sign == -1);
}

TEST_CASE("majorana_string_overlap_cancellation") {
  MajoranaString a = make_string(
      {even(0, Operator::Spin::Down), odd(0, Operator::Spin::Down), even(1, Operator::Spin::Down)});
  MajoranaString b = make_string(
      {even(0, Operator::Spin::Down), odd(0, Operator::Spin::Down), even(1, Operator::Spin::Down)});

  auto result = multiply_strings(a, b);

  CHECK(result.string.data.empty());
  // gamma_1 gamma_3 gamma_5 * gamma_1 gamma_3 gamma_5
  // b[0]=1 matches a[0], passes 2 remaining a (even) => no flip (still +1)
  // b[1]=3 matches a[1], passes 1 remaining a (odd)  => flip   (now -1)
  // b[2]=5 matches a[2], passes 0 remaining a (even) => no flip (still -1)
  CHECK(result.sign == -1);
}

TEST_CASE("majorana_string_partial_overlap") {
  MajoranaString a = make_string({even(0, Operator::Spin::Down), odd(0, Operator::Spin::Down)});
  MajoranaString b = make_string({odd(0, Operator::Spin::Up), odd(0, Operator::Spin::Down)});

  auto result = multiply_strings(a, b);

  MajoranaString expected =
      make_string({even(0, Operator::Spin::Down), odd(0, Operator::Spin::Up)});
  CHECK(result.string == expected);
  // Step by step:
  // a[0]=1 < b[0]=2 => push 1, i=1
  // a[1]=3 > b[0]=2 => b[0]=2 passes 1 remaining a-element => sign *= -1 (now -1), push 2, j=1
  // a[1]=3 == b[1]=3 => match, passes 0 remaining a-elements => no flip, cancel pair, i=2, j=2
  CHECK(result.sign == -1);
}

TEST_CASE("majorana_string_self_multiply_single_index") {
  MajoranaString a = make_string({odd(1, Operator::Spin::Down)});

  auto result = multiply_strings(a, a);

  CHECK(result.string.data.empty());
  CHECK(result.sign == 1);
}

TEST_CASE("majorana_string_identity_multiply") {
  MajoranaString a = make_string(
      {odd(0, Operator::Spin::Up), even(1, Operator::Spin::Down), even(2, Operator::Spin::Down)});
  MajoranaString empty;

  auto result_left = multiply_strings(empty, a);
  CHECK(result_left.string == a);
  CHECK(result_left.sign == 1);

  auto result_right = multiply_strings(a, empty);
  CHECK(result_right.string == a);
  CHECK(result_right.sign == 1);
}

TEST_CASE("majorana_string_anticommutation_sign") {
  // gamma_i * gamma_j = -gamma_j * gamma_i for i != j
  MajoranaString a = make_string({even(0, Operator::Spin::Up)});
  MajoranaString b = make_string({even(0, Operator::Spin::Down)});

  auto ab = multiply_strings(a, b);
  auto ba = multiply_strings(b, a);

  CHECK(ab.string == ba.string);
  CHECK(ab.sign == -ba.sign);
}

TEST_CASE("majorana_string_three_element_anticommutation") {
  // Verify sign consistency for longer strings
  MajoranaString a = make_string({even(0, Operator::Spin::Up), even(0, Operator::Spin::Down)});
  MajoranaString b = make_string({odd(0, Operator::Spin::Up), odd(0, Operator::Spin::Down)});

  auto ab = multiply_strings(a, b);
  auto ba = multiply_strings(b, a);

  MajoranaString expected = make_string({even(0, Operator::Spin::Up), even(0, Operator::Spin::Down),
                                         odd(0, Operator::Spin::Up), odd(0, Operator::Spin::Down)});
  CHECK(ab.string == expected);
  CHECK(ba.string == expected);
  // ab: b passes 2 a-elements each: 2+2=4 swaps => sign +1
  // ba: a passes 2 b-elements each: 2+2=4 swaps => sign +1
  CHECK(ab.sign == ba.sign);
}

TEST_CASE("majorana_element_accessors") {
  auto even_elem = MajoranaElement::even(3, Operator::Spin::Up);
  auto odd_elem = MajoranaElement::odd(2, Operator::Spin::Down);

  CHECK(even_elem.orbital() == 3u);
  CHECK(even_elem.spin() == Operator::Spin::Up);
  CHECK(even_elem.parity() == MajoranaElement::Parity::Even);
  CHECK(even_elem.is_even());
  CHECK(!even_elem.is_odd());

  CHECK(odd_elem.orbital() == 2u);
  CHECK(odd_elem.spin() == Operator::Spin::Down);
  CHECK(odd_elem.parity() == MajoranaElement::Parity::Odd);
  CHECK(!odd_elem.is_even());
  CHECK(odd_elem.is_odd());
}

TEST_CASE("majorana_element_ordering_by_packed_bits") {
  auto e_u0 = MajoranaElement::even(0, Operator::Spin::Up);
  auto e_d0 = MajoranaElement::even(0, Operator::Spin::Down);
  auto o_u0 = MajoranaElement::odd(0, Operator::Spin::Up);
  auto o_d0 = MajoranaElement::odd(0, Operator::Spin::Down);
  auto e_u1 = MajoranaElement::even(1, Operator::Spin::Up);

  CHECK(e_u0 < e_d0);
  CHECK(e_d0 < o_u0);
  CHECK(o_u0 < o_d0);
  CHECK(o_d0 < e_u1);
}

}  // namespace majorana_string_tests
