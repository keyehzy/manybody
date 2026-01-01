#include "static_vector.h"

#include <array>

#include "framework.h"
#include "operator.h"

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

TEST(static_vector_equality) {
  Operator a = Operator::creation(Operator::Spin::Up, 1);
  Operator b = Operator::annihilation(Operator::Spin::Down, 2);

  static_vector<Operator, 4> lhs({a, b});
  static_vector<Operator, 4> rhs({a, b});
  static_vector<Operator, 4> different({b});

  EXPECT_TRUE(lhs == rhs);
  EXPECT_TRUE(!(lhs == different));
}
