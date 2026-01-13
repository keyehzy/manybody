#include <stdexcept>

#include "framework.h"
#include "utils/index.h"

TEST(dynamic_index_round_trip) {
  DynamicIndex index({2, 3, 4});

  EXPECT_EQ(index.size(), 24u);

  const std::vector<DynamicIndex::size_type> coordinates{1, 2, 3};
  EXPECT_EQ(index.to_orbital(coordinates), 23u);

  const auto recovered = index.from_orbital(23);
  EXPECT_EQ(recovered[0], 1u);
  EXPECT_EQ(recovered[1], 2u);
  EXPECT_EQ(recovered[2], 3u);

  EXPECT_EQ(index.value_at(23, 1), 2u);
}

TEST(dynamic_index_bounds_checks) {
  DynamicIndex index({2, 2});

  bool threw = false;
  try {
    (void)index.to_orbital({1, 2});
  } catch (const std::out_of_range&) {
    threw = true;
  }
  EXPECT_TRUE(threw);

  threw = false;
  try {
    (void)index.to_orbital({1});
  } catch (const std::out_of_range&) {
    threw = true;
  }
  EXPECT_TRUE(threw);

  threw = false;
  try {
    (void)index.from_orbital(4);
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
    DynamicIndex index({2, 0});
    (void)index.size();
  } catch (const std::out_of_range&) {
    threw = true;
  }
  EXPECT_TRUE(threw);
}

TEST(dynamic_index_operator_overloads) {
  DynamicIndex index({3, 4});

  const std::vector<DynamicIndex::size_type> coordinates{2, 1};
  EXPECT_EQ(index(coordinates), index.to_orbital(coordinates));
  EXPECT_EQ(index({2, 1}), index.to_orbital({2, 1}));

  const auto round_trip = index(5);
  EXPECT_EQ(round_trip[0], 2u);
  EXPECT_EQ(round_trip[1], 1u);
}

TEST(dynamic_index_wrap_offsets) {
  DynamicIndex index({4, 3});

  const std::vector<DynamicIndex::size_type> coordinates{0, 2};
  const std::vector<int> offsets{-1, 2};
  const auto expected = index.to_orbital({3, 1});
  EXPECT_EQ(index(coordinates, offsets), expected);
  EXPECT_EQ(index({0, 2}, {-1, 2}), index.to_orbital({3, 1}));
}

TEST(dynamic_index_wrap_bounds_checks) {
  DynamicIndex index({2, 2});

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
