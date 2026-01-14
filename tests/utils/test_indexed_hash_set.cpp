#include <stdexcept>

#include "catch.hpp"
#include "utils/indexed_hash_set.h"

TEST_CASE("indexed_hash_set_initializer_list_preserves_order") {
  IndexedHashSet<int> set{3, 1, 4};

  CHECK((set.size()) == (3u));
  CHECK((set[0]) == (3));
  CHECK((set[1]) == (1));
  CHECK((set[2]) == (4));
  CHECK(set.contains(1));
}

TEST_CASE("indexed_hash_set_index_of_and_elements") {
  IndexedHashSet<int> set{10, 20, 30};

  CHECK((set.index_of(20)) == (1u));
  CHECK((set.elements()[2]) == (30));
}

TEST_CASE("indexed_hash_set_at_out_of_range_throws") {
  IndexedHashSet<int> set{1};
  bool threw = false;
  try {
    set.at(2);
  } catch (const std::out_of_range&) {
    threw = true;
  }

  CHECK(threw);
}

TEST_CASE("indexed_hash_set_index_of_missing_throws") {
  IndexedHashSet<int> set{1, 2};
  bool threw = false;
  try {
    set.index_of(4);
  } catch (const std::out_of_range&) {
    threw = true;
  }

  CHECK(threw);
}

TEST_CASE("indexed_hash_set_equality_uses_order") {
  IndexedHashSet<int> a{1, 2, 3};
  IndexedHashSet<int> b{1, 2, 3};
  IndexedHashSet<int> c{3, 2, 1};

  CHECK(a == b);
  CHECK(a != c);
}
