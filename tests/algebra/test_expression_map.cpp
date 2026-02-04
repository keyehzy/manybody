#include <catch2/catch.hpp>

#include "algebra/expression_map.h"
#include "algebra/term.h"
#include "utils/tolerances.h"

using MapType = ExpressionMap<Term::container_type>;
using complex_type = MapType::complex_type;
using key_type = MapType::key_type;

TEST_CASE("expression_map_is_zero_below_tolerance") {
  constexpr auto tol = tolerances::tolerance<double>();
  CHECK(MapType::is_zero(complex_type(0.0, 0.0)));
  CHECK(MapType::is_zero(complex_type(0.5 * tol, 0.0)));
  CHECK(MapType::is_zero(complex_type(0.0, 0.5 * tol)));
  CHECK_FALSE(MapType::is_zero(complex_type(1.0, 0.0)));
  CHECK_FALSE(MapType::is_zero(complex_type(0.0, 1.0)));
}

TEST_CASE("expression_map_add_inserts_new_key") {
  MapType m;
  key_type key{Operator::creation(Operator::Spin::Up, 0)};
  m.add(key, complex_type(1.0, 0.0));
  CHECK(m.size() == 1u);
  auto it = m.data.find(key);
  CHECK(it != m.data.end());
  CHECK(it->second == complex_type(1.0, 0.0));
}

TEST_CASE("expression_map_add_combines_existing_key") {
  MapType m;
  key_type key{Operator::creation(Operator::Spin::Up, 0)};
  m.add(key, complex_type(1.0, 0.0));
  m.add(key, complex_type(2.0, 0.0));
  CHECK(m.size() == 1u);
  CHECK(m.data[key] == complex_type(3.0, 0.0));
}

TEST_CASE("expression_map_add_prunes_zero") {
  MapType m;
  key_type key{Operator::creation(Operator::Spin::Up, 0)};
  m.add(key, complex_type(1.0, 0.0));
  m.add(key, complex_type(-1.0, 0.0));
  CHECK(m.size() == 0u);
}

TEST_CASE("expression_map_add_ignores_zero_coeff") {
  MapType m;
  key_type key{Operator::creation(Operator::Spin::Up, 0)};
  m.add(key, complex_type(0.0, 0.0));
  CHECK(m.size() == 0u);
}

TEST_CASE("expression_map_add_scalar") {
  MapType m;
  m.add_scalar(complex_type(2.0, 1.0));
  CHECK(m.size() == 1u);
  auto it = m.data.find(key_type{});
  CHECK(it != m.data.end());
  CHECK(it->second == complex_type(2.0, 1.0));
}

TEST_CASE("expression_map_subtract_scalar") {
  MapType m;
  m.add_scalar(complex_type(2.0, 1.0));
  m.subtract_scalar(complex_type(2.0, 1.0));
  CHECK(m.size() == 0u);
}

TEST_CASE("expression_map_scale") {
  MapType m;
  key_type key{Operator::creation(Operator::Spin::Up, 0)};
  m.add(key, complex_type(2.0, 0.0));
  m.scale(complex_type(3.0, 0.0));
  CHECK(m.data[key] == complex_type(6.0, 0.0));
}

TEST_CASE("expression_map_scale_by_zero_clears") {
  MapType m;
  m.add(key_type{Operator::creation(Operator::Spin::Up, 0)}, complex_type(2.0, 0.0));
  m.scale(complex_type(0.0, 0.0));
  CHECK(m.empty());
}

TEST_CASE("expression_map_divide") {
  MapType m;
  key_type key{Operator::creation(Operator::Spin::Up, 0)};
  m.add(key, complex_type(6.0, 0.0));
  m.divide(complex_type(3.0, 0.0));
  CHECK(m.data[key] == complex_type(2.0, 0.0));
}

TEST_CASE("expression_map_add_all") {
  MapType a;
  MapType b;
  key_type k1{Operator::creation(Operator::Spin::Up, 0)};
  key_type k2{Operator::creation(Operator::Spin::Down, 1)};
  a.add(k1, complex_type(1.0, 0.0));
  b.add(k1, complex_type(2.0, 0.0));
  b.add(k2, complex_type(3.0, 0.0));
  a.add_all(b);
  CHECK(a.size() == 2u);
  CHECK(a.data[k1] == complex_type(3.0, 0.0));
  CHECK(a.data[k2] == complex_type(3.0, 0.0));
}

TEST_CASE("expression_map_subtract_all") {
  MapType a;
  MapType b;
  key_type k1{Operator::creation(Operator::Spin::Up, 0)};
  a.add(k1, complex_type(3.0, 0.0));
  b.add(k1, complex_type(3.0, 0.0));
  a.subtract_all(b);
  CHECK(a.empty());
}

TEST_CASE("expression_map_truncate_by_norm") {
  MapType m;
  key_type k1{Operator::creation(Operator::Spin::Up, 0)};
  key_type k2{Operator::creation(Operator::Spin::Down, 1)};
  m.add(k1, complex_type(0.1, 0.0));
  m.add(k2, complex_type(2.0, 0.0));
  m.truncate_by_norm(0.5);
  CHECK(m.size() == 1u);
  CHECK(m.data.find(k2) != m.data.end());
}

TEST_CASE("expression_map_truncate_by_norm_zero_is_noop") {
  MapType m;
  m.add(key_type{}, complex_type(0.1, 0.0));
  m.truncate_by_norm(0.0);
  CHECK(m.size() == 1u);
}

TEST_CASE("expression_map_format_sorted_empty") {
  MapType m;
  std::ostringstream oss;
  m.format_sorted(oss,
                  [](std::ostringstream& os, const key_type&, const complex_type&) { os << "x"; });
  CHECK(oss.str() == "0");
}

TEST_CASE("expression_map_format_sorted_orders_by_size_then_norm") {
  MapType m;
  Operator a = Operator::creation(Operator::Spin::Up, 0);
  Operator b = Operator::creation(Operator::Spin::Up, 1);
  key_type k_empty{};
  key_type k_one{a};
  key_type k_two{a, b};

  m.add(k_two, complex_type(5.0, 0.0));
  m.add(k_empty, complex_type(1.0, 0.0));
  m.add(k_one, complex_type(3.0, 0.0));

  std::vector<size_t> sizes;
  std::ostringstream oss;
  m.format_sorted(oss,
                  [&sizes](std::ostringstream& os, const key_type& key, const complex_type& c) {
                    sizes.push_back(key.size());
                    os << std::abs(c);
                  });

  REQUIRE(sizes.size() == 3u);
  CHECK(sizes[0] == 0u);
  CHECK(sizes[1] == 1u);
  CHECK(sizes[2] == 2u);
}

TEST_CASE("expression_map_empty_map_operations") {
  MapType m;
  CHECK(m.empty());
  CHECK(m.size() == 0u);
  m.scale(complex_type(2.0, 0.0));
  CHECK(m.empty());
  m.truncate_by_norm(1.0);
  CHECK(m.empty());
  MapType other;
  m.add_all(other);
  CHECK(m.empty());
}
