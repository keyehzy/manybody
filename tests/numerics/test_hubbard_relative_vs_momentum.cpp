// Cross-checks of the two-particle relative-coordinate operators against the momentum-space
// Hubbard model, restricted to the sector |p up, K-p down> with fixed total momentum K.
//
// The relative operators act on the symmetric-gauge wavefunction phi(r), with
//   Psi(x_up, x_down) = exp(i K (x_up + x_down) / 2) phi(x_up - x_down),
// so the map from momentum amplitudes a_p to phi is
//   phi(r) = (1/sqrt N) sum_p a_p exp(i (2 pi p / L - pi K / L) . r).
// Periodicity in x_up makes phi(r + L_d e_d) = (-1)^{K_d} phi(r): odd K needs an antiperiodic
// boundary in the relative coordinate, which these tests exercise.

#include <armadillo>
#include <catch2/catch.hpp>
#include <cmath>
#include <complex>
#include <cstdint>
#include <numbers>
#include <vector>

#include "algebra/fermion/basis.h"
#include "algebra/fermion/model/hubbard_model_momentum.h"
#include "algebra/fermion/model/hubbard_model_relative.h"
#include "algebra/matrix_elements.h"
#include "numerics/hubbard_relative_operators.h"
#include "utils/index.h"

namespace {

constexpr double kTolerance = 1e-10;
constexpr double kPi = std::numbers::pi_v<double>;

std::vector<std::vector<size_t>> all_momenta(const std::vector<size_t>& size) {
  const Index index(size);
  std::vector<std::vector<size_t>> momenta;
  for (size_t k = 0; k < index.size(); ++k) momenta.push_back(index(k));
  return momenta;
}

std::vector<int64_t> to_signed(const std::vector<size_t>& v) {
  return std::vector<int64_t>(v.begin(), v.end());
}

std::vector<size_t> add_momenta(const std::vector<size_t>& a, const std::vector<size_t>& b,
                                const std::vector<size_t>& size) {
  std::vector<size_t> out(size.size());
  for (size_t d = 0; d < size.size(); ++d) out[d] = (a[d] + b[d]) % size[d];
  return out;
}

FermionBasis momentum_sector(const std::vector<size_t>& size, const std::vector<size_t>& K) {
  const Index index(size);
  return FermionBasis::with_fixed_particle_number_spin_momentum(index.size(), 2, 0, index, K);
}

/// Columns: momentum basis states expressed in the relative-coordinate basis.
arma::cx_mat relative_from_momentum(const std::vector<size_t>& size, const std::vector<size_t>& K,
                                    const FermionBasis& basis) {
  const Index index(size);
  const size_t N = index.size();
  arma::cx_mat U(N, N, arma::fill::zeros);
  for (size_t i = 0; i < basis.set.size(); ++i) {
    const auto& key = basis.set[i];
    REQUIRE(key.size() == 2u);
    const bool up_first = key[0].spin() == FermionOperator::Spin::Up;
    const auto p = index(up_first ? key[0].value() : key[1].value());
    const double sign = up_first ? 1.0 : -1.0;  // reference state is c+_{p up} c+_{K-p down}
    for (size_t r = 0; r < N; ++r) {
      const auto rc = index(r);
      double phase = 0.0;
      for (size_t d = 0; d < size.size(); ++d) {
        phase += (2.0 * kPi * static_cast<double>(p[d]) - kPi * static_cast<double>(K[d])) *
                 static_cast<double>(rc[d]) / static_cast<double>(size[d]);
      }
      U(r, i) = sign * std::polar(1.0, phase) / std::sqrt(static_cast<double>(N));
    }
  }
  return U;
}

template <typename Op>
arma::cx_mat to_matrix(const Op& op) {
  const size_t N = op.dimension();
  arma::cx_mat M(N, N, arma::fill::zeros);
  for (size_t j = 0; j < N; ++j) {
    arma::cx_vec e(N, arma::fill::zeros);
    e(j) = 1.0;
    M.col(j) = op.apply(e);
  }
  return M;
}

/// Bond-centred current J_d(Q) = sum_{k,sigma} 2t sin(2 pi (k_d + Q_d/2) / L_d) c+_{k+Q} c_k.
FermionExpression bond_current(const std::vector<size_t>& size, double t,
                               const std::vector<size_t>& Q, size_t direction) {
  const Index index(size);
  FermionExpression J;
  for (size_t k = 0; k < index.size(); ++k) {
    const auto kc = index(k);
    const size_t kq = index(add_momenta(kc, Q, size));
    const double v =
        2.0 * t *
        std::sin(2.0 * kPi *
                 (static_cast<double>(kc[direction]) + 0.5 * static_cast<double>(Q[direction])) /
                 static_cast<double>(size[direction]));
    for (auto spin : {FermionOperator::Spin::Up, FermionOperator::Spin::Down}) {
      J += FermionExpression(FermionMonomial(
          FermionExpression::complex_type(v, 0.0),
          {FermionOperator::creation(spin, kq), FermionOperator::annihilation(spin, k)}));
    }
  }
  return J;
}

const std::vector<std::vector<size_t>> kLattices = {{2}, {3}, {4}, {5}, {6}, {3, 2}, {2, 3}};

}  // namespace

TEST_CASE("hubbard_relative_matches_momentum_model_for_every_total_momentum") {
  const double t = 0.8;
  const double U = 3.0;
  for (const auto& size : kLattices) {
    const HubbardModelMomentum model(t, U, size);
    const FermionExpression H = model.hamiltonian();
    for (const auto& K : all_momenta(size)) {
      CAPTURE(size, K);
      const FermionBasis basis = momentum_sector(size, K);
      const arma::cx_mat Umat = relative_from_momentum(size, K, basis);
      const arma::cx_mat H_mom = compute_matrix_elements<arma::cx_mat>(basis, H);
      const arma::cx_mat H_rel = to_matrix(HubbardRelative(size, to_signed(K), t, U));
      CHECK(arma::norm(Umat.t() * H_rel * Umat - H_mom, "inf") < kTolerance);
    }
  }
}

TEST_CASE("hubbard_relative_current_matches_momentum_current") {
  const double t = 0.8;
  for (const auto& size : kLattices) {
    for (size_t direction = 0; direction < size.size(); ++direction) {
      const std::vector<size_t> zero(size.size(), 0);
      const FermionExpression J = bond_current(size, t, zero, direction);
      for (const auto& K : all_momenta(size)) {
        CAPTURE(size, K, direction);
        const FermionBasis basis = momentum_sector(size, K);
        const arma::cx_mat Umat = relative_from_momentum(size, K, basis);
        const arma::cx_mat J_mom = compute_matrix_elements<arma::cx_mat>(basis, J);
        const arma::cx_mat J_rel =
            to_matrix(HubbardRelativeCurrent(size, to_signed(K), t, direction));
        CHECK(arma::norm(Umat.t() * J_rel * Umat - J_mom, "inf") < kTolerance);
      }
    }
  }
}

TEST_CASE("current_relative_q_matches_momentum_current_between_sectors") {
  // J(Q) maps sector K to the canonical sector (K + Q) mod L, including when K + Q wraps.
  const double t = 0.8;
  for (const auto& size : kLattices) {
    for (size_t direction = 0; direction < size.size(); ++direction) {
      for (const auto& Q : all_momenta(size)) {
        const FermionExpression J = bond_current(size, t, Q, direction);
        for (const auto& K : all_momenta(size)) {
          CAPTURE(size, K, Q, direction);
          const auto KQ = add_momenta(K, Q, size);
          const FermionBasis col_basis = momentum_sector(size, K);
          const FermionBasis row_basis = momentum_sector(size, KQ);
          const arma::cx_mat J_mom =
              compute_rectangular_matrix_elements<arma::cx_mat>(row_basis, col_basis, J);
          const CurrentRelative_Q j_q(size, t, to_signed(K), to_signed(Q), direction);
          const CurrentRelative_Q_Adjoint j_q_adj(size, t, to_signed(K), to_signed(Q), direction);
          const arma::cx_mat J_rel = to_matrix(j_q);

          const arma::cx_mat U_col = relative_from_momentum(size, K, col_basis);
          const arma::cx_mat U_row = relative_from_momentum(size, KQ, row_basis);
          CHECK(arma::norm(U_row.t() * J_rel * U_col - J_mom, "inf") < kTolerance);
          CHECK(arma::norm(to_matrix(j_q_adj) - J_rel.t(), "inf") < kTolerance);
        }
      }
    }
  }
}
