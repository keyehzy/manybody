#include <catch2/catch.hpp>

#include "algebra/majorana/majorana_string.h"

TEST_CASE("majorana_string_disjoint_multiply_concatenates") {
  MajoranaString a{0, 2, 4};
  MajoranaString b{1, 3, 5};

  auto result = multiply_strings(a, b);

  MajoranaString expected{0, 1, 2, 3, 4, 5};
  CHECK(result.string == expected);
  // b[0]=1 passes 2 a-elements (2,4), b[1]=3 passes 1 (4), b[2]=5 passes 0 => 3 swaps (odd)
  CHECK(result.sign == -1);
}

TEST_CASE("majorana_string_disjoint_multiply_odd_sign") {
  MajoranaString a{0, 2};
  MajoranaString b{1};

  auto result = multiply_strings(a, b);

  MajoranaString expected{0, 1, 2};
  CHECK(result.string == expected);
  // b[0]=1 passes 1 remaining a-element (a[1]=2) => 1 swap (odd)
  CHECK(result.sign == -1);
}

TEST_CASE("majorana_string_overlap_cancellation") {
  MajoranaString a{1, 3, 5};
  MajoranaString b{1, 3, 5};

  auto result = multiply_strings(a, b);

  CHECK(result.string.empty());
  // gamma_1 gamma_3 gamma_5 * gamma_1 gamma_3 gamma_5
  // b[0]=1 matches a[0], passes 2 remaining a (even) => no flip (still +1)
  // b[1]=3 matches a[1], passes 1 remaining a (odd)  => flip   (now -1)
  // b[2]=5 matches a[2], passes 0 remaining a (even) => no flip (still -1)
  CHECK(result.sign == -1);
}

TEST_CASE("majorana_string_partial_overlap") {
  MajoranaString a{1, 3};
  MajoranaString b{2, 3};

  auto result = multiply_strings(a, b);

  MajoranaString expected{1, 2};
  CHECK(result.string == expected);
  // Step by step:
  // a[0]=1 < b[0]=2 => push 1, i=1
  // a[1]=3 > b[0]=2 => b[0]=2 passes 1 remaining a-element => sign *= -1 (now -1), push 2, j=1
  // a[1]=3 == b[1]=3 => match, passes 0 remaining a-elements => no flip, cancel pair, i=2, j=2
  CHECK(result.sign == -1);
}

TEST_CASE("majorana_string_self_multiply_single_index") {
  MajoranaString a{7};

  auto result = multiply_strings(a, a);

  CHECK(result.string.empty());
  CHECK(result.sign == 1);
}

TEST_CASE("majorana_string_identity_multiply") {
  MajoranaString a{2, 5, 9};
  MajoranaString empty{};

  auto result_left = multiply_strings(empty, a);
  CHECK(result_left.string == a);
  CHECK(result_left.sign == 1);

  auto result_right = multiply_strings(a, empty);
  CHECK(result_right.string == a);
  CHECK(result_right.sign == 1);
}

TEST_CASE("majorana_string_anticommutation_sign") {
  // gamma_i * gamma_j = -gamma_j * gamma_i for i != j
  MajoranaString a{0};
  MajoranaString b{1};

  auto ab = multiply_strings(a, b);
  auto ba = multiply_strings(b, a);

  CHECK(ab.string == ba.string);
  CHECK(ab.sign == -ba.sign);
}

TEST_CASE("majorana_string_three_element_anticommutation") {
  // Verify sign consistency for longer strings
  MajoranaString a{0, 1};
  MajoranaString b{2, 3};

  auto ab = multiply_strings(a, b);
  auto ba = multiply_strings(b, a);

  MajoranaString expected{0, 1, 2, 3};
  CHECK(ab.string == expected);
  CHECK(ba.string == expected);
  // ab: b passes 2 a-elements each: 2+2=4 swaps => sign +1
  // ba: a passes 2 b-elements each: 2+2=4 swaps => sign +1
  CHECK(ab.sign == ba.sign);
}

TEST_CASE("majorana_index_encoding_up_spin") {
  auto op = Operator::creation(Operator::Spin::Up, 3);
  auto [even, odd] = majorana_indices(op);
  CHECK(even == 4 * 3 + 0);  // 12
  CHECK(odd == 4 * 3 + 2);   // 14
}

TEST_CASE("majorana_index_encoding_down_spin") {
  auto op = Operator::annihilation(Operator::Spin::Down, 2);
  auto [even, odd] = majorana_indices(op);
  CHECK(even == 4 * 2 + 1);  // 9
  CHECK(odd == 4 * 2 + 3);   // 11
}

TEST_CASE("majorana_index_encoding_type_independent") {
  auto create = Operator::creation(Operator::Spin::Up, 5);
  auto annihilate = Operator::annihilation(Operator::Spin::Up, 5);

  auto [ce, co] = majorana_indices(create);
  auto [ae, ao] = majorana_indices(annihilate);

  CHECK(ce == ae);
  CHECK(co == ao);
}
