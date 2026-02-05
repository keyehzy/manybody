#include <algorithm>
#include <catch2/catch.hpp>

#include "algebra/fermion/term.h"
#include "algorithms/diagonal_terms.h"

namespace {

bool contains_term(const std::vector<FermionMonomial>& terms, const FermionMonomial& value) {
  return std::find(terms.begin(), terms.end(), value) != terms.end();
}

}  // namespace

TEST_CASE("diagonal_children_assigns_off_diagonal_to_parents") {
  const auto c0 = Operator::creation(Operator::Spin::Up, 0);
  const auto c1 = Operator::creation(Operator::Spin::Up, 1);
  const auto a0 = Operator::annihilation(Operator::Spin::Up, 0);
  const auto a1 = Operator::annihilation(Operator::Spin::Up, 1);

  FermionMonomial diag_one(FermionMonomial::complex_type(1.0f, 0.0f), {c0, c1, a0, a1});
  FermionMonomial diag_two(FermionMonomial::complex_type(1.0f, 0.0f), {c0, c0, a0, a0});
  FermionMonomial off_diag(FermionMonomial::complex_type(1.0f, 0.0f), {c0, c0, a0, a1});

  Expression expr({diag_one, diag_two, off_diag});
  const auto result = group_diagonal_children(expr);

  CHECK(contains_term(result.diagonals, diag_one));
  CHECK(contains_term(result.diagonals, diag_two));

  auto it_one = result.children.find(diag_one.operators);
  CHECK(it_one != result.children.end());
  CHECK(contains_term(it_one->second, off_diag));

  auto it_two = result.children.find(diag_two.operators);
  CHECK(it_two != result.children.end());
  CHECK(contains_term(it_two->second, off_diag));
}

TEST_CASE("diagonal_children_keeps_empty_entries_for_diagonals") {
  FermionMonomial diag_term = density_density(Operator::Spin::Up, 0, Operator::Spin::Down, 1);
  FermionMonomial off_diag = one_body(Operator::Spin::Up, 0, Operator::Spin::Down, 2);

  Expression expr({diag_term, off_diag});
  const auto result = group_diagonal_children(expr);

  auto it = result.children.find(diag_term.operators);
  CHECK(it != result.children.end());
  CHECK(it->second.empty());
}

TEST_CASE("diagonal_children_matches_operator_substrings") {
  const auto c0 = Operator::creation(Operator::Spin::Up, 0);
  const auto c1 = Operator::creation(Operator::Spin::Up, 1);
  const auto c2 = Operator::creation(Operator::Spin::Up, 2);
  const auto c3 = Operator::creation(Operator::Spin::Up, 3);
  const auto a0 = Operator::annihilation(Operator::Spin::Up, 0);
  const auto a1 = Operator::annihilation(Operator::Spin::Up, 1);
  const auto a2 = Operator::annihilation(Operator::Spin::Up, 2);
  const auto a3 = Operator::annihilation(Operator::Spin::Up, 3);

  FermionMonomial diag(FermionMonomial::complex_type(1.0f, 0.0f), {c0, c2, a0, a2});
  FermionMonomial off_prefix(FermionMonomial::complex_type(1.0f, 0.0f), {c0, c2, a1, a3});
  FermionMonomial off_suffix(FermionMonomial::complex_type(1.0f, 0.0f), {c1, c3, a0, a2});
  FermionMonomial off_none(FermionMonomial::complex_type(1.0f, 0.0f), {c1, c3, a1, a3});

  Expression expr({diag, off_prefix, off_suffix, off_none});
  const auto result = group_diagonal_children(expr);

  auto it = result.children.find(diag.operators);
  CHECK(it != result.children.end());
  CHECK(contains_term(it->second, off_prefix));
  CHECK(contains_term(it->second, off_suffix));
  CHECK(!contains_term(it->second, off_none));
}

TEST_CASE("find_matching_terms_uses_term_containment") {
  const auto c0 = Operator::creation(Operator::Spin::Up, 0);
  const auto c1 = Operator::creation(Operator::Spin::Up, 1);
  const auto c2 = Operator::creation(Operator::Spin::Up, 2);
  const auto a0 = Operator::annihilation(Operator::Spin::Up, 0);
  const auto a1 = Operator::annihilation(Operator::Spin::Up, 1);
  const auto a2 = Operator::annihilation(Operator::Spin::Up, 2);

  FermionMonomial reference(FermionMonomial::complex_type(1.0f, 0.0f), {c0, c1});
  FermionMonomial exact(FermionMonomial::complex_type(2.0f, 0.0f), {c0, c1});
  FermionMonomial contains_prefix(FermionMonomial::complex_type(1.0f, 0.0f), {c0, c1, a0, a1});
  FermionMonomial contains_suffix(FermionMonomial::complex_type(1.0f, 0.0f), {a0, a1, c0, c1});
  FermionMonomial contains_middle(FermionMonomial::complex_type(1.0f, 0.0f), {a0, c0, c1, a1});
  FermionMonomial none(FermionMonomial::complex_type(1.0f, 0.0f), {c2, a2});

  Expression expr({exact, contains_prefix, contains_suffix, contains_middle, none});
  const auto matches = find_matching_terms(reference, expr);

  CHECK(contains_term(matches, exact));
  CHECK(contains_term(matches, contains_prefix));
  CHECK(contains_term(matches, contains_suffix));
  CHECK(contains_term(matches, contains_middle));
  CHECK(!contains_term(matches, none));
}

TEST_CASE("find_matching_terms_handles_single_operator") {
  const auto c0 = Operator::creation(Operator::Spin::Up, 0);
  const auto c1 = Operator::creation(Operator::Spin::Up, 1);
  const auto a0 = Operator::annihilation(Operator::Spin::Up, 0);

  FermionMonomial reference(FermionMonomial::complex_type(1.0f, 0.0f), {c0});
  FermionMonomial exact(FermionMonomial::complex_type(1.0f, 0.0f), {c0});
  FermionMonomial embedded(FermionMonomial::complex_type(1.0f, 0.0f), {c0, a0});
  FermionMonomial other(FermionMonomial::complex_type(1.0f, 0.0f), {c1});

  Expression expr({exact, embedded, other});
  const auto matches = find_matching_terms(reference, expr);

  CHECK(contains_term(matches, exact));
  CHECK(contains_term(matches, embedded));
  CHECK(!contains_term(matches, other));
}

TEST_CASE("find_matching_terms_allows_out_of_order_matching") {
  const auto c0 = Operator::creation(Operator::Spin::Up, 0);
  const auto c1 = Operator::creation(Operator::Spin::Up, 1);
  const auto a0 = Operator::annihilation(Operator::Spin::Up, 0);
  const auto a1 = Operator::annihilation(Operator::Spin::Up, 1);
  const auto c2 = Operator::creation(Operator::Spin::Up, 2);
  const auto a2 = Operator::annihilation(Operator::Spin::Up, 2);

  FermionMonomial reference(FermionMonomial::complex_type(1.0f, 0.0f), {c0, c1});
  FermionMonomial contiguous_match(FermionMonomial::complex_type(1.0f, 0.0f), {a2, c0, c1, a0, a1});
  FermionMonomial non_contiguous(FermionMonomial::complex_type(1.0f, 0.0f), {c0, c2, c1, a2});
  FermionMonomial missing_operator(FermionMonomial::complex_type(1.0f, 0.0f), {c0, c2, a2});

  Expression expr({contiguous_match, non_contiguous, missing_operator});
  const auto matches = find_matching_terms(reference, expr);

  CHECK(contains_term(matches, contiguous_match));
  CHECK(contains_term(matches, non_contiguous));
  CHECK(!contains_term(matches, missing_operator));
}

TEST_CASE("find_matching_terms_empty_reference_returns_empty") {
  const auto c0 = Operator::creation(Operator::Spin::Up, 0);
  const auto a0 = Operator::annihilation(Operator::Spin::Up, 0);

  FermionMonomial empty_reference;
  FermionMonomial density_term(FermionMonomial::complex_type(1.0f, 0.0f), {c0, a0});

  Expression expr({density_term});
  const auto matches = find_matching_terms(empty_reference, expr);

  CHECK(matches.empty());
}
