#include <algorithm>

#include "algebra/term.h"
#include "algorithms/diagonal_terms.h"
#include "framework.h"

namespace {

bool contains_term(const std::vector<Term>& terms, const Term& value) {
  return std::find(terms.begin(), terms.end(), value) != terms.end();
}

}  // namespace

TEST(diagonal_children_assigns_off_diagonal_to_parents) {
  const auto c0 = Operator::creation(Operator::Spin::Up, 0);
  const auto c1 = Operator::creation(Operator::Spin::Up, 1);
  const auto a0 = Operator::annihilation(Operator::Spin::Up, 0);
  const auto a1 = Operator::annihilation(Operator::Spin::Up, 1);

  Term diag_one(Term::complex_type(1.0f, 0.0f), {c0, c1, a0, a1});
  Term diag_two(Term::complex_type(1.0f, 0.0f), {c0, c0, a0, a0});
  Term off_diag(Term::complex_type(1.0f, 0.0f), {c0, c0, a0, a1});

  Expression expr({diag_one, diag_two, off_diag});
  const auto result = group_diagonal_children(expr);

  EXPECT_TRUE(contains_term(result.diagonals, diag_one));
  EXPECT_TRUE(contains_term(result.diagonals, diag_two));

  auto it_one = result.children.find(diag_one.operators);
  EXPECT_TRUE(it_one != result.children.end());
  EXPECT_TRUE(contains_term(it_one->second, off_diag));

  auto it_two = result.children.find(diag_two.operators);
  EXPECT_TRUE(it_two != result.children.end());
  EXPECT_TRUE(contains_term(it_two->second, off_diag));
}

TEST(diagonal_children_keeps_empty_entries_for_diagonals) {
  Term diag_term = density_density(Operator::Spin::Up, 0, Operator::Spin::Down, 1);
  Term off_diag = one_body(Operator::Spin::Up, 0, Operator::Spin::Down, 2);

  Expression expr({diag_term, off_diag});
  const auto result = group_diagonal_children(expr);

  auto it = result.children.find(diag_term.operators);
  EXPECT_TRUE(it != result.children.end());
  EXPECT_TRUE(it->second.empty());
}

TEST(diagonal_children_matches_operator_substrings) {
  const auto c0 = Operator::creation(Operator::Spin::Up, 0);
  const auto c1 = Operator::creation(Operator::Spin::Up, 1);
  const auto c2 = Operator::creation(Operator::Spin::Up, 2);
  const auto c3 = Operator::creation(Operator::Spin::Up, 3);
  const auto a0 = Operator::annihilation(Operator::Spin::Up, 0);
  const auto a1 = Operator::annihilation(Operator::Spin::Up, 1);
  const auto a2 = Operator::annihilation(Operator::Spin::Up, 2);
  const auto a3 = Operator::annihilation(Operator::Spin::Up, 3);

  Term diag(Term::complex_type(1.0f, 0.0f), {c0, c2, a0, a2});
  Term off_prefix(Term::complex_type(1.0f, 0.0f), {c0, c2, a1, a3});
  Term off_suffix(Term::complex_type(1.0f, 0.0f), {c1, c3, a0, a2});
  Term off_none(Term::complex_type(1.0f, 0.0f), {c1, c3, a1, a3});

  Expression expr({diag, off_prefix, off_suffix, off_none});
  const auto result = group_diagonal_children(expr);

  auto it = result.children.find(diag.operators);
  EXPECT_TRUE(it != result.children.end());
  EXPECT_TRUE(contains_term(it->second, off_prefix));
  EXPECT_TRUE(contains_term(it->second, off_suffix));
  EXPECT_TRUE(!contains_term(it->second, off_none));
}

TEST(find_matching_terms_uses_term_substrings) {
  const auto c0 = Operator::creation(Operator::Spin::Up, 0);
  const auto c1 = Operator::creation(Operator::Spin::Up, 1);
  const auto c2 = Operator::creation(Operator::Spin::Up, 2);
  const auto c3 = Operator::creation(Operator::Spin::Up, 3);
  const auto a0 = Operator::annihilation(Operator::Spin::Up, 0);
  const auto a1 = Operator::annihilation(Operator::Spin::Up, 1);
  const auto a2 = Operator::annihilation(Operator::Spin::Up, 2);
  const auto a3 = Operator::annihilation(Operator::Spin::Up, 3);

  Term reference(Term::complex_type(1.0f, 0.0f), {c0, c1, a0, a1});
  Term same(Term::complex_type(2.0f, 0.0f), {c0, c1, a0, a1});
  Term prefix(Term::complex_type(1.0f, 0.0f), {c0, c1, a2, a3});
  Term suffix(Term::complex_type(1.0f, 0.0f), {c2, c3, a0, a1});
  Term none(Term::complex_type(1.0f, 0.0f), {c2, c3, a2, a3});

  Expression expr({same, prefix, suffix, none});
  const auto matches = find_matching_terms(reference, expr);

  EXPECT_TRUE(contains_term(matches, same));
  EXPECT_TRUE(contains_term(matches, prefix));
  EXPECT_TRUE(contains_term(matches, suffix));
  EXPECT_TRUE(!contains_term(matches, none));
}

TEST(find_matching_terms_handles_single_operator) {
  const auto c0 = Operator::creation(Operator::Spin::Up, 0);
  const auto c1 = Operator::creation(Operator::Spin::Up, 1);
  const auto a0 = Operator::annihilation(Operator::Spin::Up, 0);

  Term reference(Term::complex_type(1.0f, 0.0f), {c0});
  Term exact(Term::complex_type(1.0f, 0.0f), {c0});
  Term embedded(Term::complex_type(1.0f, 0.0f), {c0, a0});
  Term other(Term::complex_type(1.0f, 0.0f), {c1});

  Expression expr({exact, embedded, other});
  const auto matches = find_matching_terms(reference, expr);

  EXPECT_TRUE(contains_term(matches, exact));
  EXPECT_TRUE(contains_term(matches, embedded));
  EXPECT_TRUE(!contains_term(matches, other));
}

TEST(find_matching_terms_requires_contiguous_substring) {
  const auto c0 = Operator::creation(Operator::Spin::Up, 0);
  const auto c1 = Operator::creation(Operator::Spin::Up, 1);
  const auto a0 = Operator::annihilation(Operator::Spin::Up, 0);
  const auto a1 = Operator::annihilation(Operator::Spin::Up, 1);
  const auto c2 = Operator::creation(Operator::Spin::Up, 2);
  const auto a2 = Operator::annihilation(Operator::Spin::Up, 2);

  Term reference(Term::complex_type(1.0f, 0.0f), {c0, c1, a0, a1});
  Term contiguous_match(Term::complex_type(1.0f, 0.0f), {c0, c1, a2, a0, a1});
  Term non_contiguous(Term::complex_type(1.0f, 0.0f), {c0, c2, a0, a2, c1, a1});

  Expression expr({contiguous_match, non_contiguous});
  const auto matches = find_matching_terms(reference, expr);

  EXPECT_TRUE(contains_term(matches, contiguous_match));
  EXPECT_TRUE(!contains_term(matches, non_contiguous));
}

TEST(find_matching_terms_empty_reference_returns_empty) {
  const auto c0 = Operator::creation(Operator::Spin::Up, 0);
  const auto a0 = Operator::annihilation(Operator::Spin::Up, 0);

  Term empty_reference;
  Term density_term(Term::complex_type(1.0f, 0.0f), {c0, a0});

  Expression expr({density_term});
  const auto matches = find_matching_terms(empty_reference, expr);

  EXPECT_TRUE(matches.empty());
}
