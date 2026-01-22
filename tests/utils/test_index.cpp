#include <stdexcept>

#include <catch2/catch.hpp>
#include "utils/index.h"

TEST_CASE("dynamic_index_round_trip") {
  Index index({2, 3, 4});

  CHECK((index.size()) == (24u));

  const std::vector<Index::size_type> coordinates{1, 2, 3};
  CHECK((index(coordinates)) == (23u));

  const auto recovered = index(23);
  CHECK((recovered[0]) == (1u));
  CHECK((recovered[1]) == (2u));
  CHECK((recovered[2]) == (3u));

  CHECK((index.value_at(23, 1)) == (2u));
}

TEST_CASE("dynamic_index_bounds_checks") {
  Index index({2, 2});

  bool threw = false;
  try {
    (void)index({1, 2});
  } catch (const std::out_of_range&) {
    threw = true;
  }
  CHECK(threw);

  threw = false;
  try {
    (void)index({1});
  } catch (const std::out_of_range&) {
    threw = true;
  }
  CHECK(threw);

  threw = false;
  try {
    (void)index(4);
  } catch (const std::out_of_range&) {
    threw = true;
  }
  CHECK(threw);

  threw = false;
  try {
    index.dimension(3);
  } catch (const std::out_of_range&) {
    threw = true;
  }
  CHECK(threw);
}

TEST_CASE("dynamic_index_zero_dimension_throws") {
  bool threw = false;
  try {
    Index index({2, 0});
    (void)index.size();
  } catch (const std::out_of_range&) {
    threw = true;
  }
  CHECK(threw);
}

TEST_CASE("dynamic_index_operator_overloads") {
  Index index({3, 4});

  const std::vector<Index::size_type> coordinates{2, 1};
  CHECK((index(coordinates)) == (index(coordinates)));
  CHECK((index({2, 1})) == (index({2, 1})));

  const auto round_trip = index(5);
  CHECK((round_trip[0]) == (2u));
  CHECK((round_trip[1]) == (1u));
}

TEST_CASE("dynamic_index_wrap_offsets") {
  Index index({4, 3});

  const std::vector<Index::size_type> coordinates{0, 2};
  const std::vector<int> offsets{-1, 2};
  const auto expected = index({3, 1});
  CHECK((index(coordinates, offsets)) == (expected));
  CHECK((index({0, 2}, {-1, 2})) == (index({3, 1})));
}

TEST_CASE("dynamic_index_wrap_bounds_checks") {
  Index index({2, 2});

  bool threw = false;
  try {
    (void)index({1, 0, 0}, {0, 0});
  } catch (const std::out_of_range&) {
    threw = true;
  }
  CHECK(threw);

  threw = false;
  try {
    (void)index({1, 0}, {0});
  } catch (const std::out_of_range&) {
    threw = true;
  }
  CHECK(threw);
}
