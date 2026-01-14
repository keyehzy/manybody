#include <stdexcept>

#include "framework.h"
#include "utils/index.h"

TEST(dynamic_index_round_trip) {
  Index index({2, 3, 4});

  EXPECT_EQ(index.size(), 24u);

  const std::vector<Index::size_type> coordinates{1, 2, 3};
  EXPECT_EQ(index(coordinates), 23u);

  const auto recovered = index(23);
  EXPECT_EQ(recovered[0], 1u);
  EXPECT_EQ(recovered[1], 2u);
  EXPECT_EQ(recovered[2], 3u);

  EXPECT_EQ(index.value_at(23, 1), 2u);
}

TEST(dynamic_index_bounds_checks) {
  Index index({2, 2});

  bool threw = false;
  try {
    (void)index({1, 2});
  } catch (const std::out_of_range&) {
    threw = true;
  }
  EXPECT_TRUE(threw);

  threw = false;
  try {
    (void)index({1});
  } catch (const std::out_of_range&) {
    threw = true;
  }
  EXPECT_TRUE(threw);

  threw = false;
  try {
    (void)index(4);
  } catch (const std::out_of_range&) {
    threw = true;
  }
  EXPECT_TRUE(threw);

  threw = false;
  try {
    index.dimension(3);
  } catch (const std::out_of_range&) {
    threw = true;
  }
  EXPECT_TRUE(threw);
}

TEST(dynamic_index_zero_dimension_throws) {
  bool threw = false;
  try {
    Index index({2, 0});
    (void)index.size();
  } catch (const std::out_of_range&) {
    threw = true;
  }
  EXPECT_TRUE(threw);
}

TEST(dynamic_index_operator_overloads) {
  Index index({3, 4});

  const std::vector<Index::size_type> coordinates{2, 1};
  EXPECT_EQ(index(coordinates), index(coordinates));
  EXPECT_EQ(index({2, 1}), index({2, 1}));

  const auto round_trip = index(5);
  EXPECT_EQ(round_trip[0], 2u);
  EXPECT_EQ(round_trip[1], 1u);
}

TEST(dynamic_index_wrap_offsets) {
  Index index({4, 3});

  const std::vector<Index::size_type> coordinates{0, 2};
  const std::vector<int> offsets{-1, 2};
  const auto expected = index({3, 1});
  EXPECT_EQ(index(coordinates, offsets), expected);
  EXPECT_EQ(index({0, 2}, {-1, 2}), index({3, 1}));
}

TEST(dynamic_index_wrap_bounds_checks) {
  Index index({2, 2});

  bool threw = false;
  try {
    (void)index({1, 0, 0}, {0, 0});
  } catch (const std::out_of_range&) {
    threw = true;
  }
  EXPECT_TRUE(threw);

  threw = false;
  try {
    (void)index({1, 0}, {0});
  } catch (const std::out_of_range&) {
    threw = true;
  }
  EXPECT_TRUE(threw);
}
