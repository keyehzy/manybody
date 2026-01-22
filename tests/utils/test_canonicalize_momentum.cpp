#include <catch2/catch.hpp>

#include "utils/canonicalize_momentum.h"

TEST_CASE("canonicalize_momentum_positive_values") {
  CHECK((utils::canonicalize_momentum(0, 4)) == (0u));
  CHECK((utils::canonicalize_momentum(1, 4)) == (1u));
  CHECK((utils::canonicalize_momentum(3, 4)) == (3u));
  CHECK((utils::canonicalize_momentum(4, 4)) == (0u));
  CHECK((utils::canonicalize_momentum(5, 4)) == (1u));
  CHECK((utils::canonicalize_momentum(8, 4)) == (0u));
}

TEST_CASE("canonicalize_momentum_negative_values") {
  CHECK((utils::canonicalize_momentum(-1, 4)) == (3u));
  CHECK((utils::canonicalize_momentum(-2, 4)) == (2u));
  CHECK((utils::canonicalize_momentum(-4, 4)) == (0u));
  CHECK((utils::canonicalize_momentum(-5, 4)) == (3u));
  CHECK((utils::canonicalize_momentum(-8, 4)) == (0u));
}

TEST_CASE("canonicalize_momentum_zero_lattice_size") {
  CHECK((utils::canonicalize_momentum(0, 0)) == (0u));
  CHECK((utils::canonicalize_momentum(5, 0)) == (0u));
  CHECK((utils::canonicalize_momentum(-5, 0)) == (0u));
}

TEST_CASE("canonicalize_momentum_vector_version") {
  const std::vector<int64_t> momentum = {-1, 5, 0};
  const std::vector<size_t> size = {4, 3, 2};
  const auto result = utils::canonicalize_momentum(momentum, size);

  CHECK((result.size()) == (3u));
  CHECK((result[0]) == (3u));
  CHECK((result[1]) == (2u));
  CHECK((result[2]) == (0u));
}

TEST_CASE("canonicalize_momentum_vector_empty") {
  const std::vector<int64_t> momentum = {};
  const std::vector<size_t> size = {};
  const auto result = utils::canonicalize_momentum(momentum, size);

  CHECK(result.empty());
}
