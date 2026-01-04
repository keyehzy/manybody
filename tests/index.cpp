#include "index.h"

#include <stdexcept>

#include "framework.h"

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

TEST(dynamic_index_constructor_limits_orbitals) {
  bool threw = false;
  try {
    DynamicIndex too_large({Operator::max_index() + 1});
  } catch (const std::out_of_range&) {
    threw = true;
  }
  EXPECT_TRUE(threw);
}
