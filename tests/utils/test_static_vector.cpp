#include <array>
#include <catch2/catch.hpp>
#include <unordered_set>

#include "algebra/operator.h"
#include "utils/static_vector.h"

TEST_CASE("static_vector_default_is_empty") {
  static_vector<int, 4> vec;
  CHECK((vec.size()) == (0u));
  CHECK(vec.begin() == vec.end());
}

TEST_CASE("static_vector_initializer_list_preserves_order") {
  static_vector<int, 4> vec({1, 3, 5});
  CHECK((vec.size()) == (3u));
  auto it = vec.begin();
  CHECK((*it++) == (1));
  CHECK((*it++) == (3));
  CHECK((*it++) == (5));
}

TEST_CASE("static_vector_push_back_and_append_range") {
  static_vector<int, 5> vec;
  vec.push_back(4);
  std::array<int, 2> values{7, 9};
  vec.append_range(values.begin(), values.end());
  CHECK((vec.size()) == (3u));
  auto it = vec.begin();
  CHECK((*it++) == (4));
  CHECK((*it++) == (7));
  CHECK((*it++) == (9));
}

TEST_CASE("static_vector_reverse_iterators") {
  static_vector<int, 4> vec({2, 4, 6});
  auto it = vec.rbegin();
  CHECK((*it++) == (6));
  CHECK((*it++) == (4));
  CHECK((*it++) == (2));
}

TEST_CASE("static_vector_index_access") {
  static_vector<int, 4> vec({1, 2, 3});
  vec[1] = 5;
  CHECK((vec[0]) == (1));
  CHECK((vec[1]) == (5));
  CHECK((vec[2]) == (3));

  const static_vector<int, 4> const_vec({4, 6, 8});
  CHECK((const_vec[1]) == (6));
}

TEST_CASE("static_vector_at_access") {
  static_vector<int, 4> vec({10, 20, 30});
  vec.at(2) = 40;
  CHECK((vec.at(0)) == (10));
  CHECK((vec.at(1)) == (20));
  CHECK((vec.at(2)) == (40));

  const static_vector<int, 4> const_vec({11, 22, 33});
  CHECK((const_vec.at(2)) == (33));
}

TEST_CASE("static_vector_equality") {
  Operator a = Operator::creation(Operator::Spin::Up, 1);
  Operator b = Operator::annihilation(Operator::Spin::Down, 2);

  static_vector<Operator, 4> lhs({a, b});
  static_vector<Operator, 4> rhs({a, b});
  static_vector<Operator, 4> different({b});

  CHECK(lhs == rhs);
  CHECK(!(lhs == different));
}

TEST_CASE("static_vector_hash_matches_equal") {
  static_vector<int, 4> lhs({1, 2, 3});
  static_vector<int, 4> rhs({1, 2, 3});
  std::hash<static_vector<int, 4>> hasher;
  CHECK((hasher(lhs)) == (hasher(rhs)));
}

TEST_CASE("static_vector_hash_works_in_unordered_set") {
  std::unordered_set<static_vector<int, 4>> values;
  static_vector<int, 4> a({1, 2});
  static_vector<int, 4> b({1, 2});
  static_vector<int, 4> c({2, 1});

  values.insert(a);
  values.insert(c);

  CHECK(values.find(b) != values.end());
  CHECK(values.find(c) != values.end());
}

TEST_CASE("static_vector_insert_at_beginning") {
  static_vector<int, 5> vec({2, 3, 4});
  vec.insert(0, 1);
  CHECK((vec.size()) == (4u));
  CHECK((vec[0]) == (1));
  CHECK((vec[1]) == (2));
  CHECK((vec[2]) == (3));
  CHECK((vec[3]) == (4));
}

TEST_CASE("static_vector_insert_in_middle") {
  static_vector<int, 5> vec({1, 2, 4});
  vec.insert(2, 3);
  CHECK((vec.size()) == (4u));
  CHECK((vec[0]) == (1));
  CHECK((vec[1]) == (2));
  CHECK((vec[2]) == (3));
  CHECK((vec[3]) == (4));
}

TEST_CASE("static_vector_insert_at_end") {
  static_vector<int, 5> vec({1, 2, 3});
  vec.insert(3, 4);
  CHECK((vec.size()) == (4u));
  CHECK((vec[0]) == (1));
  CHECK((vec[1]) == (2));
  CHECK((vec[2]) == (3));
  CHECK((vec[3]) == (4));
}

TEST_CASE("static_vector_insert_into_empty") {
  static_vector<int, 5> vec;
  vec.insert(0, 42);
  CHECK((vec.size()) == (1u));
  CHECK((vec[0]) == (42));
}

TEST_CASE("static_vector_erase_at_beginning") {
  static_vector<int, 5> vec({1, 2, 3, 4});
  vec.erase(0);
  CHECK((vec.size()) == (3u));
  CHECK((vec[0]) == (2));
  CHECK((vec[1]) == (3));
  CHECK((vec[2]) == (4));
}

TEST_CASE("static_vector_erase_in_middle") {
  static_vector<int, 5> vec({1, 2, 3, 4});
  vec.erase(2);
  CHECK((vec.size()) == (3u));
  CHECK((vec[0]) == (1));
  CHECK((vec[1]) == (2));
  CHECK((vec[2]) == (4));
}

TEST_CASE("static_vector_erase_at_end") {
  static_vector<int, 5> vec({1, 2, 3, 4});
  vec.erase(3);
  CHECK((vec.size()) == (3u));
  CHECK((vec[0]) == (1));
  CHECK((vec[1]) == (2));
  CHECK((vec[2]) == (3));
}

TEST_CASE("static_vector_erase_single_element") {
  static_vector<int, 5> vec({42});
  vec.erase(0);
  CHECK((vec.size()) == (0u));
  CHECK(vec.empty());
}

TEST_CASE("static_vector_insert_and_erase_combined") {
  static_vector<int, 6> vec({1, 3, 5});

  // Insert 2 between 1 and 3
  vec.insert(1, 2);
  CHECK((vec.size()) == (4u));
  CHECK((vec[0]) == (1));
  CHECK((vec[1]) == (2));
  CHECK((vec[2]) == (3));
  CHECK((vec[3]) == (5));

  // Insert 4 between 3 and 5
  vec.insert(3, 4);
  CHECK((vec.size()) == (5u));
  CHECK((vec[3]) == (4));
  CHECK((vec[4]) == (5));

  // Erase 3 (at index 2)
  vec.erase(2);
  CHECK((vec.size()) == (4u));
  CHECK((vec[0]) == (1));
  CHECK((vec[1]) == (2));
  CHECK((vec[2]) == (4));
  CHECK((vec[3]) == (5));
}
