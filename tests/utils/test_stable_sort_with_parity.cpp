#include <algorithm>
#include <catch2/catch.hpp>
#include <vector>

#include "utils/stable_sort_with_parity.h"

namespace {

template <typename T>
using Container = std::vector<T>;

template <typename T>
bool inversion_parity(const Container<T>& values) {
  bool odd = false;
  for (std::size_t i = 0; i < values.size(); ++i) {
    for (std::size_t j = i + 1; j < values.size(); ++j) {
      if (values[j] < values[i]) {
        odd = !odd;
      }
    }
  }
  return odd;
}

struct StableItem {
  int key{};
  int id{};
};

bool operator<(const StableItem& lhs, const StableItem& rhs) { return lhs.key < rhs.key; }

}  // namespace

TEST_CASE("stable_sort_with_parity_handles_empty_and_single") {
  Container<int> empty;
  Container<int> empty_tmp;
  bool empty_odd = false;
  stable_sort_with_parity(empty, empty_tmp, 0, empty.size(), empty_odd);
  CHECK(empty.empty());
  CHECK(empty_odd == false);

  Container<int> single{5};
  Container<int> single_tmp(single.size());
  bool single_odd = false;
  stable_sort_with_parity(single, single_tmp, 0, single.size(), single_odd);
  CHECK(single[0] == 5);
  CHECK(single_odd == false);
}

TEST_CASE("stable_sort_with_parity_preserves_stability_with_equal_keys") {
  Container<StableItem> values{{2, 0}, {1, 1}, {2, 2}, {1, 3}, {2, 4}};
  const auto expected_odd = inversion_parity(values);
  Container<StableItem> tmp(values.size());
  bool odd = false;

  stable_sort_with_parity(values, tmp, 0, values.size(), odd);

  CHECK((values[0].key) == (1));
  CHECK((values[0].id) == (1));
  CHECK((values[1].key) == (1));
  CHECK((values[1].id) == (3));
  CHECK((values[2].key) == (2));
  CHECK((values[2].id) == (0));
  CHECK((values[3].key) == (2));
  CHECK((values[3].id) == (2));
  CHECK((values[4].key) == (2));
  CHECK((values[4].id) == (4));
  CHECK(odd == expected_odd);
}

TEST_CASE("stable_sort_with_parity_matches_inversion_parity_for_permutations") {
  Container<int> values{1, 2, 3, 4};

  do {
    const auto expected_odd = inversion_parity(values);
    auto data = values;
    Container<int> tmp(data.size());
    bool odd = false;

    stable_sort_with_parity(data, tmp, 0, data.size(), odd);

    CHECK(std::is_sorted(data.begin(), data.end()));
    CHECK(odd == expected_odd);
  } while (std::next_permutation(values.begin(), values.end()));
}
