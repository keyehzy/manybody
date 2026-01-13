#include <array>
#include <unordered_set>

#include "algebra/operator.h"
#include "framework.h"
#include "utils/static_vector.h"

TEST(static_vector_default_is_empty) {
  static_vector<int, 4> vec;
  EXPECT_EQ(vec.size(), 0u);
  EXPECT_TRUE(vec.begin() == vec.end());
}

TEST(static_vector_initializer_list_preserves_order) {
  static_vector<int, 4> vec({1, 3, 5});
  EXPECT_EQ(vec.size(), 3u);
  auto it = vec.begin();
  EXPECT_EQ(*it++, 1);
  EXPECT_EQ(*it++, 3);
  EXPECT_EQ(*it++, 5);
}

TEST(static_vector_push_back_and_append_range) {
  static_vector<int, 5> vec;
  vec.push_back(4);
  std::array<int, 2> values{7, 9};
  vec.append_range(values.begin(), values.end());
  EXPECT_EQ(vec.size(), 3u);
  auto it = vec.begin();
  EXPECT_EQ(*it++, 4);
  EXPECT_EQ(*it++, 7);
  EXPECT_EQ(*it++, 9);
}

TEST(static_vector_reverse_iterators) {
  static_vector<int, 4> vec({2, 4, 6});
  auto it = vec.rbegin();
  EXPECT_EQ(*it++, 6);
  EXPECT_EQ(*it++, 4);
  EXPECT_EQ(*it++, 2);
}

TEST(static_vector_index_access) {
  static_vector<int, 4> vec({1, 2, 3});
  vec[1] = 5;
  EXPECT_EQ(vec[0], 1);
  EXPECT_EQ(vec[1], 5);
  EXPECT_EQ(vec[2], 3);

  const static_vector<int, 4> const_vec({4, 6, 8});
  EXPECT_EQ(const_vec[1], 6);
}

TEST(static_vector_at_access) {
  static_vector<int, 4> vec({10, 20, 30});
  vec.at(2) = 40;
  EXPECT_EQ(vec.at(0), 10);
  EXPECT_EQ(vec.at(1), 20);
  EXPECT_EQ(vec.at(2), 40);

  const static_vector<int, 4> const_vec({11, 22, 33});
  EXPECT_EQ(const_vec.at(2), 33);
}

TEST(static_vector_equality) {
  Operator a = Operator::creation(Operator::Spin::Up, 1);
  Operator b = Operator::annihilation(Operator::Spin::Down, 2);

  static_vector<Operator, 4> lhs({a, b});
  static_vector<Operator, 4> rhs({a, b});
  static_vector<Operator, 4> different({b});

  EXPECT_TRUE(lhs == rhs);
  EXPECT_TRUE(!(lhs == different));
}

TEST(static_vector_hash_matches_equal) {
  static_vector<int, 4> lhs({1, 2, 3});
  static_vector<int, 4> rhs({1, 2, 3});
  std::hash<static_vector<int, 4>> hasher;
  EXPECT_EQ(hasher(lhs), hasher(rhs));
}

TEST(static_vector_hash_works_in_unordered_set) {
  std::unordered_set<static_vector<int, 4>> values;
  static_vector<int, 4> a({1, 2});
  static_vector<int, 4> b({1, 2});
  static_vector<int, 4> c({2, 1});

  values.insert(a);
  values.insert(c);

  EXPECT_TRUE(values.find(b) != values.end());
  EXPECT_TRUE(values.find(c) != values.end());
}
