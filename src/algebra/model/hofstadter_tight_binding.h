#pragma once

#include <cmath>
#include <complex>
#include <cstddef>
#include <numeric>
#include <stdexcept>

#include "algebra/fermion/expression.h"
#include "algebra/fermion/model/model.h"
#include "utils/index.h"

/// Non-interacting tight-binding model on a 2D square lattice in a
/// perpendicular magnetic field (Hofstadter model) with Landau gauge.
///
/// The Peierls phases use Landau gauge A = (0, Bx), so y-direction
/// hoppings carry a phase exp(i 2 pi alpha x), where alpha = p/q is the
/// flux per plaquette in units of the flux quantum.
///
/// For periodic boundaries the torus compatibility constraint requires
/// integer total flux: (Nx * Ny * p) % q == 0.  The x-boundary hopping
/// carries a gauge-compensation phase exp(-i 2 pi alpha Nx y).
///
/// Site indexing: site(x, y) = x * Ny + y  via Index({Nx, Ny}).
struct HofstadterTightBindingModel : FermionModel {
  static constexpr auto spin = FermionOperator::Spin::Up;

  HofstadterTightBindingModel(double t_val, size_t nx_val, size_t ny_val, double alpha_val)
      : t(t_val),
        nx(nx_val),
        ny(ny_val),
        alpha(alpha_val),
        num_sites(nx_val * ny_val),
        index({nx_val, ny_val}) {
    auto [p, q] = flux_fraction();
    if ((nx * ny * static_cast<size_t>(p)) % q != 0) {
      throw std::invalid_argument(
          "Periodic boundary conditions require integer total flux "
          "through the torus");
    }
  }

  size_t site(size_t x, size_t y) const { return index({x, y}); }

  /// Return the flux fraction alpha = p/q as (p, q) with q > 0, gcd = 1.
  std::pair<size_t, size_t> flux_fraction() const {
    // Represent alpha as a fraction with denominator up to max_q.
    // We scan denominators and pick the one whose rational approximation
    // is closest to alpha.
    constexpr size_t max_q = 256;
    constexpr double tol = 1e-12;
    size_t best_p = 0, best_q = 1;
    double best_err = std::abs(alpha);
    for (size_t q_try = 1; q_try <= max_q; ++q_try) {
      auto p_try = static_cast<size_t>(std::round(alpha * q_try));
      double err = std::abs(alpha - static_cast<double>(p_try) / q_try);
      if (err < best_err) {
        best_err = err;
        best_p = p_try;
        best_q = q_try;
        if (err < tol) break;
      }
    }
    if (best_err > tol) {
      throw std::invalid_argument("alpha is not close to a rational with small denominator");
    }
    size_t g = std::gcd(best_p, best_q);
    return {best_p / g, best_q / g};
  }

  /// Number of states in one magnetic subband: Nx * Ny / q.
  size_t subband_size() const {
    auto [p, q] = flux_fraction();
    return num_sites / q;
  }

  /// Number of magnetic subbands: q.
  size_t n_subbands() const {
    auto [p, q] = flux_fraction();
    return q;
  }

  FermionExpression hamiltonian() const override {
    FermionExpression result;
    using cx = FermionExpression::complex_type;
    const double two_pi_alpha = 2.0 * M_PI * alpha;

    for (size_t x = 0; x < nx; ++x) {
      for (size_t y = 0; y < ny; ++y) {
        const size_t src = site(x, y);

        // x-direction hopping (no Peierls phase for bulk bonds)
        if (x + 1 < nx) {
          result += hopping(cx(-t, 0.0), src, site(x + 1, y), spin);
        } else {
          // Wrap: gauge compensation phase on x-boundary
          double boundary_phase = -two_pi_alpha * static_cast<double>(nx * y);
          cx amp(-t * std::cos(boundary_phase), -t * std::sin(boundary_phase));
          result += hopping(amp, src, site(0, y), spin);
        }

        // y-direction hopping with Landau-gauge Peierls phase
        double peierls_angle = two_pi_alpha * static_cast<double>(x);
        cx y_amp(-t * std::cos(peierls_angle), -t * std::sin(peierls_angle));
        if (y + 1 < ny) {
          result += hopping(y_amp, src, site(x, y + 1), spin);
        } else {
          result += hopping(y_amp, src, site(x, 0), spin);
        }
      }
    }
    return result;
  }

  double t;
  size_t nx;
  size_t ny;
  double alpha;
  size_t num_sites;
  Index index;
};
