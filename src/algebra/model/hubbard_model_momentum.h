#pragma once

#include <cmath>
#include <cstddef>
#include <numbers>
#include <stdexcept>
#include <vector>

#include "algebra/model/model.h"
#include "algebra/term.h"
#include "utils/index.h"

/// Dimension-agnostic Hubbard model in momentum space.
///
/// H = sum_{k,sigma} ε(k) c†_{k,σ} c_{k,σ}
///   + (U/N) sum_{k1,k2,q} c†_{k1+q,↑} c†_{k2-q,↓} c_{k2,↓} c_{k1,↑}
///
/// where ε(k) = -2t * sum_d cos(2π k_d / L_d)
struct HubbardModelMomentum : Model {
  HubbardModelMomentum(double t_val, double u_val, const std::vector<size_t>& size_val)
      : t(t_val), u(u_val), size(size_val), index(size_val) {
    if (size.empty()) {
      throw std::invalid_argument("HubbardModelMomentum requires at least one dimension.");
    }
  }

  /// Compute the dispersion relation ε(k) = -2t * sum_d cos(2π k_d / L_d)
  double dispersion(const std::vector<size_t>& momentum) const {
    double energy = 0.0;
    for (size_t d = 0; d < size.size(); ++d) {
      const double phase = 2.0 * std::numbers::pi_v<double> * static_cast<double>(momentum[d]) /
                           static_cast<double>(size[d]);
      energy += -2.0 * t * std::cos(phase);
    }
    return energy;
  }

  /// Compute the group velocity v_d(k) = ∂ε(k)/∂k_d = 2t * (2π/L_d) * sin(2π k_d / L_d)
  double velocity(const std::vector<size_t>& momentum, size_t direction) const {
    if (direction >= size.size()) {
      throw std::invalid_argument("Direction index out of bounds.");
    }
    const double phase = 2.0 * std::numbers::pi_v<double> * static_cast<double>(momentum[direction]) /
                         static_cast<double>(size[direction]);
    return 2.0 * t * (2.0 * std::numbers::pi_v<double> / static_cast<double>(size[direction])) *
           std::sin(phase);
  }

  /// Current operator in direction d with momentum transfer Q:
  /// J_d(Q) = sum_{k,σ} v_d(k) c†_{k+Q,σ} c_{k,σ}
  Expression current(const std::vector<size_t>& Q, size_t direction) const {
    if (Q.size() != size.size()) {
      throw std::invalid_argument("Momentum transfer Q must have the same dimension as the system.");
    }
    if (direction >= size.size()) {
      throw std::invalid_argument("Direction index out of bounds.");
    }

    Expression current_term;
    const size_t N = index.size();

    for (size_t k = 0; k < N; ++k) {
      const auto k_coords = index(k);
      const double v = velocity(k_coords, direction);

      if (v == 0.0) {
        continue;
      }

      const auto coeff = Expression::complex_type(v, 0.0);

      // Compute k + Q with periodic boundary conditions
      std::vector<size_t> k_plus_Q(size.size());
      for (size_t d = 0; d < size.size(); ++d) {
        k_plus_Q[d] = (k_coords[d] + Q[d]) % size[d];
      }
      const size_t k_plus_Q_idx = index(k_plus_Q);

      // c†_{k+Q,↑} c_{k,↑}
      current_term += Expression(Term(coeff, {Operator::creation(Operator::Spin::Up, k_plus_Q_idx),
                                              Operator::annihilation(Operator::Spin::Up, k)}));
      // c†_{k+Q,↓} c_{k,↓}
      current_term += Expression(Term(coeff, {Operator::creation(Operator::Spin::Down, k_plus_Q_idx),
                                              Operator::annihilation(Operator::Spin::Down, k)}));
    }
    return current_term;
  }

  Expression kinetic() const {
    Expression kinetic_term;
    for (size_t k = 0; k < index.size(); ++k) {
      const auto momentum = index(k);
      const double energy = dispersion(momentum);
      const auto coeff = Expression::complex_type(energy, 0.0);

      // n_{k,up} = c†_{k,up} c_{k,up}
      kinetic_term += Expression(Term(coeff, {Operator::creation(Operator::Spin::Up, k),
                                              Operator::annihilation(Operator::Spin::Up, k)}));
      // n_{k,down} = c†_{k,down} c_{k,down}
      kinetic_term += Expression(Term(coeff, {Operator::creation(Operator::Spin::Down, k),
                                              Operator::annihilation(Operator::Spin::Down, k)}));
    }
    return kinetic_term;
  }

  Expression interaction() const {
    Expression interaction_term;
    const size_t N = index.size();
    const auto u_coeff = Expression::complex_type(u / static_cast<double>(N), 0.0);

    // (U/N) sum_{k1,k2,q} c†_{k1+q,↑} c†_{k2-q,↓} c_{k2,↓} c_{k1,↑}
    for (size_t k1 = 0; k1 < N; ++k1) {
      for (size_t k2 = 0; k2 < N; ++k2) {
        for (size_t q = 0; q < N; ++q) {
          // Compute k1+q and k2-q with periodic boundary conditions
          const auto k1_coords = index(k1);
          const auto k2_coords = index(k2);
          const auto q_coords = index(q);

          std::vector<size_t> k1_plus_q(size.size());
          std::vector<size_t> k2_minus_q(size.size());
          for (size_t d = 0; d < size.size(); ++d) {
            k1_plus_q[d] = (k1_coords[d] + q_coords[d]) % size[d];
            k2_minus_q[d] = (k2_coords[d] + size[d] - q_coords[d]) % size[d];
          }

          const size_t k1_plus_q_idx = index(k1_plus_q);
          const size_t k2_minus_q_idx = index(k2_minus_q);

          interaction_term +=
              Expression(Term(u_coeff, {Operator::creation(Operator::Spin::Up, k1_plus_q_idx),
                                        Operator::creation(Operator::Spin::Down, k2_minus_q_idx),
                                        Operator::annihilation(Operator::Spin::Down, k2),
                                        Operator::annihilation(Operator::Spin::Up, k1)}));
        }
      }
    }
    return interaction_term;
  }

  Expression hamiltonian() const override {
    Expression result = kinetic();
    result += interaction();
    return result;
  }

  double t;
  double u;
  std::vector<size_t> size;
  Index index;
};
