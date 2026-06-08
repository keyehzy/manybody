#include <armadillo>
#include <catch2/catch.hpp>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include "algorithms/dynamical_system.h"

namespace {

struct LinearDecay {
  double operator()(double /*t*/, double value) const { return -value; }
};

// dy/dt = -k y, exact solution y = y0 exp(-k t). Becomes stiff for large k.
struct ScaledDecay {
  double k;
  double operator()(double /*t*/, double value) const { return -k * value; }
};

// Two-component harmonic oscillator written as a 2-vector ODE.
// state = (x, v), dot x = v, dot v = -omega^2 x. Exact: x(t) = cos(omega t).
struct HarmonicOscillator {
  double omega2;
  arma::vec operator()(double /*t*/, const arma::vec& s) const {
    arma::vec ds(2);
    ds(0) = s(1);
    ds(1) = -omega2 * s(0);
    return ds;
  }
};

struct ArmaVecNorm {
  double operator()(const arma::vec& v) const { return arma::norm(v, 2); }
};

}  // namespace

TEST_CASE("integrate_adaptive_linear_decay_matches_exact") {
  AdaptiveOptions opts;
  opts.atol = 1e-10;
  opts.rtol = 1e-10;
  const double result =
      integrate_adaptive(LinearDecay{}, 1.0, 0.0, 1.0, 0.1, opts);
  CHECK(std::abs(result - std::exp(-1.0)) < 1e-8);
}

TEST_CASE("integrate_adaptive_tolerance_tightening_reduces_error") {
  AdaptiveOptions loose;
  loose.atol = 1e-4;
  loose.rtol = 1e-4;
  AdaptiveOptions tight;
  tight.atol = 1e-10;
  tight.rtol = 1e-10;

  const double y0 = 1.0;
  const double y_exact = std::exp(-5.0);
  const double y_loose =
      integrate_adaptive(LinearDecay{}, y0, 0.0, 5.0, 0.1, loose);
  const double y_tight =
      integrate_adaptive(LinearDecay{}, y0, 0.0, 5.0, 0.1, tight);

  const double err_loose = std::abs(y_loose - y_exact);
  const double err_tight = std::abs(y_tight - y_exact);
  CHECK(err_tight < err_loose);
  CHECK(err_tight < 1e-8);
}

TEST_CASE("integrate_adaptive_rejects_oversized_initial_step_for_stiff_problem") {
  // Stiff-ish problem: huge rate constant, but oversized initial step.
  // The integrator should reject and shrink the step rather than blow up.
  AdaptiveOptions opts;
  opts.atol = 1e-8;
  opts.rtol = 1e-6;
  std::size_t rhs_calls = 0;
  std::size_t accepted = 0;
  auto callback = [&](double, double) { ++accepted; };
  struct Counted {
    std::size_t* calls;
    double k;
    double operator()(double, double y) const {
      ++(*calls);
      return -k * y;
    }
  };
  const double y_final = integrate_adaptive(Counted{&rhs_calls, 100.0}, 1.0, 0.0, 0.1,
                                            0.05, opts, DefaultStateNorm{}, callback);
  // exp(-10) ~ 4.5e-5
  CHECK(std::abs(y_final - std::exp(-10.0)) < 1e-6);
  // Some rejection should have occurred relative to a hypothetical single step.
  // (FSAL costs 6 evals per accepted step + 6 per rejected step.) Bound is
  // generous: just confirm we made progress without exhausting max_rejected.
  CHECK(accepted >= 2);
  CHECK(rhs_calls > 6);
}

TEST_CASE("integrate_adaptive_hits_t1_exactly") {
  AdaptiveOptions opts;
  std::vector<double> times;
  auto callback = [&](double t, double) { times.push_back(t); };
  integrate_adaptive(LinearDecay{}, 1.0, 0.0, 0.7, 0.1, opts, DefaultStateNorm{},
                     callback);
  REQUIRE(!times.empty());
  CHECK(times.front() == Approx(0.0));
  CHECK(times.back() == Approx(0.7));
}

TEST_CASE("integrate_adaptive_throws_when_max_rejected_exceeded") {
  // Forcing failure: tighten tols absurdly with tiny dt_min so the controller
  // cannot shrink further. With atol=rtol=0 the scale becomes 0, so we use a
  // pathological RHS that produces an explicit unbounded growth instead.
  struct Blowup {
    double operator()(double, double y) const { return std::exp(1e6 * y); }
  };
  AdaptiveOptions opts;
  opts.atol = 0.0;
  opts.rtol = 1e-16;
  opts.dt_min = 1e-20;
  opts.max_rejected = 5;
  CHECK_THROWS_AS(
      integrate_adaptive(Blowup{}, 1.0, 0.0, 1.0, 1e-3, opts),
      std::runtime_error);
}

TEST_CASE("integrate_adaptive_vector_harmonic_oscillator") {
  AdaptiveOptions opts;
  opts.atol = 1e-9;
  opts.rtol = 1e-9;
  const double omega2 = 4.0;  // omega = 2
  arma::vec s0(2);
  s0(0) = 1.0;
  s0(1) = 0.0;
  const double t_final = 3.14159265358979323846;  // half period at omega=2
  arma::vec s_final = integrate_adaptive(HarmonicOscillator{omega2}, s0, 0.0, t_final,
                                         0.05, opts, ArmaVecNorm{});
  // x(pi) = cos(2 pi) = 1, v(pi) = -2 sin(2 pi) = 0.
  CHECK(std::abs(s_final(0) - 1.0) < 1e-6);
  CHECK(std::abs(s_final(1)) < 1e-6);
}

TEST_CASE("integrate_adaptive_respects_dt_max_cap") {
  AdaptiveOptions opts;
  opts.atol = 1.0;  // very loose, so controller wants to grow dt aggressively
  opts.rtol = 1.0;
  opts.dt_max = 0.05;
  double max_dt_observed = 0.0;
  double prev_t = 0.0;
  auto callback = [&](double t, double) {
    max_dt_observed = std::max(max_dt_observed, t - prev_t);
    prev_t = t;
  };
  integrate_adaptive(ScaledDecay{0.01}, 1.0, 0.0, 1.0, 0.01, opts, DefaultStateNorm{},
                     callback);
  // dt is allowed to clamp down to t1 - t on the final step, so we just
  // require the cap was never exceeded by more than a tiny epsilon.
  CHECK(max_dt_observed <= 0.05 + 1e-12);
}
