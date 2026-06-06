#include <catch2/catch.hpp>
#include <complex>

#include "algebra/fermion/expression.h"
#include "algebra/reference_state.h"
#include "algebra/wick_decouple.h"

namespace {

using complex_type = FermionExpression::complex_type;
using Spin = FermionOperator::Spin;

complex_type operator_coeff(const FermionExpression& e,
                            const FermionMonomial::container_type& key) {
  auto it = e.terms().find(key);
  return (it == e.terms().end()) ? complex_type{0.0, 0.0} : it->second;
}

bool approx_equal(complex_type a, complex_type b) { return std::norm(a - b) < 1e-20; }

DiagonalFermionReference make_diagonal_reference(std::size_t n_sites, double n_value,
                                                 double D_value) {
  return DiagonalFermionReference(n_sites, complex_type{n_value, 0.0}, complex_type{D_value, 0.0});
}

}  // namespace

TEST_CASE("vacuum_reference_recovers_canonicalize_for_c_then_cdag") {
  FermionOperator c = FermionOperator::annihilation(Spin::Down, 3);
  FermionOperator c_dag = FermionOperator::creation(Spin::Down, 3);
  FermionMonomial term(complex_type{1.0, 0.0}, {c, c_dag});

  VacuumFermionReference vac;
  FermionExpression decoupled = wick_decouple(term, vac);
  FermionExpression reference = canonicalize(term);

  CHECK(decoupled.size() == reference.size());
  for (const auto& [ops, coeff] : reference.terms()) {
    CHECK(approx_equal(operator_coeff(decoupled, ops), coeff));
  }
}

TEST_CASE("vacuum_reference_recovers_canonicalize_for_four_body") {
  FermionOperator a_dag = FermionOperator::creation(Spin::Up, 1);
  FermionOperator a = FermionOperator::annihilation(Spin::Up, 1);
  FermionOperator b_dag = FermionOperator::creation(Spin::Down, 2);
  FermionOperator b = FermionOperator::annihilation(Spin::Down, 2);
  FermionMonomial term(complex_type{1.0, 0.0}, {a, a_dag, b, b_dag});

  VacuumFermionReference vac;
  FermionExpression decoupled = wick_decouple(term, vac);
  FermionExpression reference = canonicalize(term);

  CHECK(decoupled.size() == reference.size());
  for (const auto& [ops, coeff] : reference.terms()) {
    CHECK(approx_equal(operator_coeff(decoupled, ops), coeff));
  }
}

TEST_CASE("expectation_of_density_returns_n_k") {
  const std::size_t n_sites = 4;
  const double n_val = 0.3;
  DiagonalFermionReference ref = make_diagonal_reference(n_sites, n_val, 0.0);
  FermionMonomial term(complex_type{1.0, 0.0}, {FermionOperator::creation(Spin::Up, 2),
                                                FermionOperator::annihilation(Spin::Up, 2)});

  complex_type value = wick_expectation(term, ref);
  CHECK(approx_equal(value, complex_type{n_val, 0.0}));
}

TEST_CASE("expectation_of_hole_density_returns_one_minus_n_k") {
  const std::size_t n_sites = 4;
  const double n_val = 0.3;
  DiagonalFermionReference ref = make_diagonal_reference(n_sites, n_val, 0.0);
  FermionMonomial term(complex_type{1.0, 0.0}, {FermionOperator::annihilation(Spin::Up, 2),
                                                FermionOperator::creation(Spin::Up, 2)});

  complex_type value = wick_expectation(term, ref);
  CHECK(approx_equal(value, complex_type{1.0 - n_val, 0.0}));
}

TEST_CASE("expectation_of_off_diagonal_density_vanishes") {
  const std::size_t n_sites = 4;
  DiagonalFermionReference ref = make_diagonal_reference(n_sites, 0.5, 0.0);
  FermionMonomial term(complex_type{1.0, 0.0}, {FermionOperator::creation(Spin::Up, 1),
                                                FermionOperator::annihilation(Spin::Up, 3)});

  complex_type value = wick_expectation(term, ref);
  CHECK(approx_equal(value, complex_type{0.0, 0.0}));
}

TEST_CASE("gaussian_wick_four_point_decomposition_same_spin") {
  const std::size_t n_sites = 8;
  DiagonalFermionReference ref = make_diagonal_reference(n_sites, 0.0, 0.0);
  ref.n[1][static_cast<std::size_t>(Spin::Up)] = complex_type{0.7, 0.0};
  ref.n[5][static_cast<std::size_t>(Spin::Up)] = complex_type{0.4, 0.0};

  // <c+_1 c_1 c+_5 c_5> = <c+_1 c_1> <c+_5 c_5> - <c+_1 c_5> <c+_5 c_1>
  //                    = n_1 n_5 - 0
  FermionMonomial term(
      complex_type{1.0, 0.0},
      {FermionOperator::creation(Spin::Up, 1), FermionOperator::annihilation(Spin::Up, 1),
       FermionOperator::creation(Spin::Up, 5), FermionOperator::annihilation(Spin::Up, 5)});

  complex_type value = wick_expectation(term, ref);
  CHECK(approx_equal(value, complex_type{0.7 * 0.4, 0.0}));
}

TEST_CASE("gaussian_wick_four_point_decomposition_exchange") {
  const std::size_t n_sites = 8;
  DiagonalFermionReference ref = make_diagonal_reference(n_sites, 0.0, 0.0);
  ref.n[1][static_cast<std::size_t>(Spin::Up)] = complex_type{0.6, 0.0};
  ref.n[5][static_cast<std::size_t>(Spin::Up)] = complex_type{0.5, 0.0};

  // <c+_1 c_5 c+_5 c_1> with the diagonal-density reference:
  //   contractions: (0,3) <c+_1 c_1> = n_1, sign for j=3 -> (-1)^2 = +1
  //                 (1,2) <c_5 c+_5> = 1 - n_5
  //   so the (0,3)x(1,2) full pairing contributes (+1) * n_1 * (1 - n_5).
  // No other full pairing survives. wick_expectation should return n_1 (1 - n_5).
  FermionMonomial term(
      complex_type{1.0, 0.0},
      {FermionOperator::creation(Spin::Up, 1), FermionOperator::annihilation(Spin::Up, 5),
       FermionOperator::creation(Spin::Up, 5), FermionOperator::annihilation(Spin::Up, 1)});

  complex_type value = wick_expectation(term, ref);
  CHECK(approx_equal(value, complex_type{0.6 * (1.0 - 0.5), 0.0}));
}

TEST_CASE("on_site_double_occupancy_uses_D_cumulant") {
  const std::size_t n_sites = 2;
  const double n_val = 0.5;
  const double D_val = 0.1;  // D < n_up n_dn for a correlated suppression.
  DiagonalFermionReference ref = make_diagonal_reference(n_sites, n_val, D_val);

  // n_{0,up} n_{0,down} = c+_{0,up} c_{0,up} c+_{0,down} c_{0,down}.
  FermionMonomial term(
      complex_type{1.0, 0.0},
      {FermionOperator::creation(Spin::Up, 0), FermionOperator::annihilation(Spin::Up, 0),
       FermionOperator::creation(Spin::Down, 0), FermionOperator::annihilation(Spin::Down, 0)});

  complex_type value = wick_expectation(term, ref);
  CHECK(approx_equal(value, complex_type{D_val, 0.0}));
}

TEST_CASE("on_site_double_occupancy_off_site_vanishing_correction") {
  const std::size_t n_sites = 2;
  const double n_val = 0.5;
  const double D_val = 0.1;
  DiagonalFermionReference ref = make_diagonal_reference(n_sites, n_val, D_val);

  // n_{0,up} n_{1,down}: different sites, no cumulant injection.
  FermionMonomial term(
      complex_type{1.0, 0.0},
      {FermionOperator::creation(Spin::Up, 0), FermionOperator::annihilation(Spin::Up, 0),
       FermionOperator::creation(Spin::Down, 1), FermionOperator::annihilation(Spin::Down, 1)});

  complex_type value = wick_expectation(term, ref);
  CHECK(approx_equal(value, complex_type{n_val * n_val, 0.0}));
}

TEST_CASE("decouple_six_body_produces_four_and_two_body_partial_contractions") {
  const std::size_t n_sites = 4;
  DiagonalFermionReference ref = make_diagonal_reference(n_sites, 0.5, 0.0);

  FermionMonomial term(
      complex_type{1.0, 0.0},
      {FermionOperator::creation(Spin::Up, 0), FermionOperator::creation(Spin::Down, 1),
       FermionOperator::creation(Spin::Up, 2), FermionOperator::annihilation(Spin::Up, 2),
       FermionOperator::annihilation(Spin::Down, 1), FermionOperator::annihilation(Spin::Up, 0)});

  FermionExpression decoupled = wick_decouple(term, ref);

  std::size_t scalar_count = 0;
  std::size_t two_body = 0;
  std::size_t four_body = 0;
  std::size_t six_body = 0;
  for (const auto& [ops, coeff] : decoupled.terms()) {
    const std::size_t s = ops.size();
    if (s == 0)
      ++scalar_count;
    else if (s == 2)
      ++two_body;
    else if (s == 4)
      ++four_body;
    else if (s == 6)
      ++six_body;
  }

  CHECK(scalar_count >= 1);
  CHECK(two_body >= 1);
  CHECK(four_body >= 1);
  CHECK(six_body >= 1);
}

TEST_CASE("decouple_preserves_canonicalize_on_2body_for_vacuum") {
  // Sanity: for any string, sum over Wick subsets against vacuum should give
  // the same operator as plain canonicalize().
  FermionOperator a = FermionOperator::annihilation(Spin::Up, 4);
  FermionOperator b = FermionOperator::annihilation(Spin::Down, 2);
  FermionOperator c = FermionOperator::creation(Spin::Up, 4);
  FermionOperator d = FermionOperator::creation(Spin::Down, 2);
  FermionMonomial term(complex_type{1.0, 0.0}, {a, b, c, d});

  VacuumFermionReference vac;
  FermionExpression decoupled = wick_decouple(term, vac);
  FermionExpression reference = canonicalize(term);

  CHECK(decoupled.size() == reference.size());
  for (const auto& [ops, coeff] : reference.terms()) {
    CHECK(approx_equal(operator_coeff(decoupled, ops), coeff));
  }
}

TEST_CASE("wick_expectation_distributes_over_expression") {
  const std::size_t n_sites = 4;
  DiagonalFermionReference ref = make_diagonal_reference(n_sites, 0.25, 0.0);

  FermionExpression e;
  e += FermionExpression(FermionMonomial(
      complex_type{2.0, 0.0},
      {FermionOperator::creation(Spin::Up, 1), FermionOperator::annihilation(Spin::Up, 1)}));
  e += FermionExpression(FermionMonomial(
      complex_type{-1.0, 0.0},
      {FermionOperator::creation(Spin::Down, 3), FermionOperator::annihilation(Spin::Down, 3)}));

  complex_type value = wick_expectation(e, ref);
  CHECK(approx_equal(value, complex_type{2.0 * 0.25 - 1.0 * 0.25, 0.0}));
}

TEST_CASE("general_reference_with_diagonal_rho_matches_diagonal_reference") {
  const std::size_t n_sites = 4;
  const double n_val = 0.3;

  DiagonalFermionReference diag = make_diagonal_reference(n_sites, n_val, 0.0);
  GeneralFermionReference gen(n_sites, diag.n);

  FermionMonomial term(
      complex_type{1.0, 0.0},
      {FermionOperator::creation(Spin::Up, 1), FermionOperator::annihilation(Spin::Up, 1),
       FermionOperator::creation(Spin::Down, 2), FermionOperator::annihilation(Spin::Down, 2)});

  complex_type diag_value = wick_expectation(term, diag);
  complex_type gen_value = wick_expectation(term, gen);
  CHECK(approx_equal(gen_value, diag_value));
}

TEST_CASE("general_reference_off_diagonal_density_gives_off_diagonal_expectation") {
  const std::size_t n_sites = 2;
  GeneralFermionReference gen(n_sites);
  const complex_type alpha{0.4, 0.2};
  const std::size_t i0 = gen.flat_index(Spin::Up, 0);
  const std::size_t i1 = gen.flat_index(Spin::Up, 1);
  gen.rho_at(i0, i1) = alpha;
  gen.rho_at(i1, i0) = std::conj(alpha);

  FermionMonomial term(complex_type{1.0, 0.0}, {FermionOperator::creation(Spin::Up, 0),
                                                FermionOperator::annihilation(Spin::Up, 1)});

  complex_type value = wick_expectation(term, gen);
  CHECK(approx_equal(value, alpha));
}

TEST_CASE("general_reference_particle_hole_identity") {
  const std::size_t n_sites = 2;
  GeneralFermionReference gen(n_sites);
  gen.rho_at(gen.flat_index(Spin::Up, 0), gen.flat_index(Spin::Up, 0)) = complex_type{0.6, 0.0};
  gen.rho_at(gen.flat_index(Spin::Up, 1), gen.flat_index(Spin::Up, 1)) = complex_type{0.2, 0.0};
  const complex_type alpha{0.3, -0.1};
  gen.rho_at(gen.flat_index(Spin::Up, 0), gen.flat_index(Spin::Up, 1)) = alpha;
  gen.rho_at(gen.flat_index(Spin::Up, 1), gen.flat_index(Spin::Up, 0)) = std::conj(alpha);

  FermionOperator a = FermionOperator::annihilation(Spin::Up, 0);
  FermionOperator b_dag = FermionOperator::creation(Spin::Up, 1);
  complex_type cd_dag = gen.contract_plain_then_dagger(a, b_dag);
  // <c_a c+_b> = delta_{ab} - rho_{b,a} = 0 - conj(alpha)
  CHECK(approx_equal(cd_dag, -std::conj(alpha)));

  FermionOperator c = FermionOperator::annihilation(Spin::Up, 0);
  FermionOperator c_dag = FermionOperator::creation(Spin::Up, 0);
  complex_type same = gen.contract_plain_then_dagger(c, c_dag);
  // <c_a c+_a> = 1 - rho_{a,a}
  CHECK(approx_equal(same, complex_type{1.0 - 0.6, 0.0}));
}

TEST_CASE("general_reference_four_point_wick_with_off_diagonal_rho") {
  const std::size_t n_sites = 2;
  GeneralFermionReference gen(n_sites);
  const double n0 = 0.7;
  const double n1 = 0.3;
  const complex_type alpha{0.25, 0.0};
  gen.rho_at(gen.flat_index(Spin::Up, 0), gen.flat_index(Spin::Up, 0)) = complex_type{n0, 0.0};
  gen.rho_at(gen.flat_index(Spin::Up, 1), gen.flat_index(Spin::Up, 1)) = complex_type{n1, 0.0};
  gen.rho_at(gen.flat_index(Spin::Up, 0), gen.flat_index(Spin::Up, 1)) = alpha;
  gen.rho_at(gen.flat_index(Spin::Up, 1), gen.flat_index(Spin::Up, 0)) = std::conj(alpha);

  // <c+_0 c_1 c+_1 c_0> with ops at positions 0..3 being (c+_0, c_1, c+_1, c_0).
  // Same-spin full pairings (c+, c) and their contributions:
  //   pair (0,1) and (2,3): rho_{0,1} * rho_{1,0} = |alpha|^2 (both signs +1)
  //   pair (0,3) and (1,2): rho_{0,0} * <c_1 c+_1> = n_0 * (1 - n_1) (both signs +1)
  FermionMonomial term(
      complex_type{1.0, 0.0},
      {FermionOperator::creation(Spin::Up, 0), FermionOperator::annihilation(Spin::Up, 1),
       FermionOperator::creation(Spin::Up, 1), FermionOperator::annihilation(Spin::Up, 0)});

  complex_type value = wick_expectation(term, gen);
  const complex_type expected = alpha * std::conj(alpha) + complex_type{n0 * (1.0 - n1), 0.0};
  CHECK(approx_equal(value, expected));
}

TEST_CASE("nambu_reference_zero_kappa_matches_general") {
  const std::size_t n_sites = 4;
  const double n_val = 0.3;
  DiagonalFermionReference diag = make_diagonal_reference(n_sites, n_val, 0.0);
  GeneralFermionReference gen(n_sites, diag.n);
  NambuFermionReference nambu(gen);

  FermionMonomial term(
      complex_type{1.0, 0.0},
      {FermionOperator::creation(Spin::Up, 1), FermionOperator::annihilation(Spin::Up, 1),
       FermionOperator::creation(Spin::Down, 2), FermionOperator::annihilation(Spin::Down, 2)});

  complex_type gen_value = wick_expectation(term, gen);
  complex_type nambu_value = wick_expectation(term, nambu);
  CHECK(approx_equal(nambu_value, gen_value));
}

TEST_CASE("nambu_pair_amplitude_from_kappa") {
  const std::size_t n_sites = 1;
  NambuFermionReference nambu(n_sites);
  const complex_type alpha{0.4, 0.2};
  nambu.set_pair_amplitude(Spin::Up, 0, Spin::Down, 0, alpha);

  FermionOperator c_up = FermionOperator::annihilation(Spin::Up, 0);
  FermionOperator c_dn = FermionOperator::annihilation(Spin::Down, 0);
  FermionOperator cd_up = FermionOperator::creation(Spin::Up, 0);
  FermionOperator cd_dn = FermionOperator::creation(Spin::Down, 0);

  CHECK(approx_equal(nambu.contract_plain_then_plain(c_up, c_dn), alpha));
  CHECK(approx_equal(nambu.contract_dagger_then_dagger(cd_up, cd_dn), -std::conj(alpha)));
  CHECK(approx_equal(nambu.contract_dagger_then_dagger(cd_dn, cd_up), std::conj(alpha)));
}

TEST_CASE("nambu_kappa_antisymmetry") {
  const std::size_t n_sites = 1;
  NambuFermionReference nambu(n_sites);
  const complex_type alpha{0.5, -0.3};
  nambu.set_pair_amplitude(Spin::Up, 0, Spin::Down, 0, alpha);

  FermionOperator c_up = FermionOperator::annihilation(Spin::Up, 0);
  FermionOperator c_dn = FermionOperator::annihilation(Spin::Down, 0);

  CHECK(approx_equal(nambu.contract_plain_then_plain(c_up, c_dn), alpha));
  CHECK(approx_equal(nambu.contract_plain_then_plain(c_dn, c_up), -alpha));
}

TEST_CASE("nambu_bcs_four_point_expectation") {
  // <c+_up c+_dn c_dn c_up> in a BCS-like reference with diagonal densities
  // n_up, n_dn and on-site singlet pair amplitude alpha = <c_up c_dn>.
  //   Full pairings:
  //     (0,1) c+_up c+_dn paired, (2,3) c_dn c_up paired ->
  //         <c+_up c+_dn> * <c_dn c_up> = (-alpha*) * (-alpha) = |alpha|^2
  //     (0,3) c+_up c_up paired, (1,2) c+_dn c_dn paired ->
  //         <c+_up c_up> * <c+_dn c_dn> = n_up * n_dn
  //     (0,2) c+_up c_dn ->  0 (different spins, diagonal rho).
  const std::size_t n_sites = 1;
  NambuFermionReference nambu(n_sites);
  const double n_up = 0.6;
  const double n_dn = 0.4;
  const complex_type alpha{0.3, 0.0};
  nambu.set_density(Spin::Up, 0, complex_type{n_up, 0.0});
  nambu.set_density(Spin::Down, 0, complex_type{n_dn, 0.0});
  nambu.set_pair_amplitude(Spin::Up, 0, Spin::Down, 0, alpha);

  FermionMonomial term(
      complex_type{1.0, 0.0},
      {FermionOperator::creation(Spin::Up, 0), FermionOperator::creation(Spin::Down, 0),
       FermionOperator::annihilation(Spin::Down, 0), FermionOperator::annihilation(Spin::Up, 0)});

  complex_type value = wick_expectation(term, nambu);
  const complex_type expected = complex_type{n_up * n_dn, 0.0} + alpha * std::conj(alpha);
  CHECK(approx_equal(value, expected));
}

TEST_CASE("vacuum_and_diagonal_anomalous_contractions_are_zero") {
  VacuumFermionReference vac;
  DiagonalFermionReference diag = make_diagonal_reference(2, 0.5, 0.1);

  FermionOperator c_up = FermionOperator::annihilation(Spin::Up, 0);
  FermionOperator c_dn = FermionOperator::annihilation(Spin::Down, 0);
  FermionOperator cd_up = FermionOperator::creation(Spin::Up, 0);
  FermionOperator cd_dn = FermionOperator::creation(Spin::Down, 0);

  CHECK(approx_equal(vac.contract_plain_then_plain(c_up, c_dn), complex_type{0.0, 0.0}));
  CHECK(approx_equal(vac.contract_dagger_then_dagger(cd_up, cd_dn), complex_type{0.0, 0.0}));
  CHECK(approx_equal(diag.contract_plain_then_plain(c_up, c_dn), complex_type{0.0, 0.0}));
  CHECK(approx_equal(diag.contract_dagger_then_dagger(cd_up, cd_dn), complex_type{0.0, 0.0}));
}
