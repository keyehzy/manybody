#include <stdexcept>

#include "framework.h"
#include "utils/indexed_hash_set.h"

TEST(indexed_hash_set_initializer_list_preserves_order) {
  IndexedHashSet<int> set{3, 1, 4};

  EXPECT_EQ(set.size(), 3u);
  EXPECT_EQ(set[0], 3);
  EXPECT_EQ(set[1], 1);
  EXPECT_EQ(set[2], 4);
  EXPECT_TRUE(set.contains(1));
}

TEST(indexed_hash_set_index_of_and_elements) {
  IndexedHashSet<int> set{10, 20, 30};

  EXPECT_EQ(set.index_of(20), 1u);
  EXPECT_EQ(set.elements()[2], 30);
}

TEST(indexed_hash_set_at_out_of_range_throws) {
  IndexedHashSet<int> set{1};
  bool threw = false;
  try {
    set.at(2);
  } catch (const std::out_of_range&) {
    threw = true;
  }

  EXPECT_TRUE(threw);
}

TEST(indexed_hash_set_index_of_missing_throws) {
  IndexedHashSet<int> set{1, 2};
  bool threw = false;
  try {
    set.index_of(4);
  } catch (const std::out_of_range&) {
    threw = true;
  }

  EXPECT_TRUE(threw);
}

TEST(indexed_hash_set_equality_uses_order) {
  IndexedHashSet<int> a{1, 2, 3};
  IndexedHashSet<int> b{1, 2, 3};
  IndexedHashSet<int> c{3, 2, 1};

  EXPECT_TRUE(a == b);
  EXPECT_TRUE(a != c);
}
