#include "algorithm/diagonal_terms.h"

#include <algorithm>

#include "framework.h"
#include "term.h"

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
