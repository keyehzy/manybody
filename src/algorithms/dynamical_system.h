#pragma once

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <utility>

enum class IntegratorMethod {
  kEuler,
  kRungeKutta4,
};

struct NoopCallback {
  template <typename... Args>
  void operator()(Args&&...) const {}
};

template <typename System, typename State>
State euler_step(const System& system, double t, double dt, const State& state) {
  return state + dt * system(t, state);
}

template <typename System, typename State>
State rk4_step(const System& system, double t, double dt, const State& state) {
  const State k1 = system(t, state);
  const State k2 = system(t + 0.5 * dt, state + (0.5 * dt) * k1);
  const State k3 = system(t + 0.5 * dt, state + (0.5 * dt) * k2);
  const State k4 = system(t + dt, state + dt * k3);
  return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
}

template <typename System, typename State, typename Callback>
State integrate(const System& system, State state, double t0, double t1, double dt,
                IntegratorMethod method, Callback callback) {
  if (dt <= 0.0 || t1 <= t0) {
    callback(t0, state);
    return state;
  }

  double t = t0;
  callback(t, state);
  while (t < t1) {
    const double step = std::min(dt, t1 - t);
    switch (method) {
      case IntegratorMethod::kEuler:
        state = euler_step(system, t, step, state);
        break;
      case IntegratorMethod::kRungeKutta4:
        state = rk4_step(system, t, step, state);
        break;
      default:
        break;
    }
    t += step;
    callback(t, state);
  }
  return state;
}

template <typename System, typename State>
State integrate(const System& system, State state, double t0, double t1, double dt,
                IntegratorMethod method) {
  return integrate(system, std::move(state), t0, t1, dt, method, NoopCallback{});
}

// =====================================================================
// Adaptive Dormand-Prince 5(4) ("RK45") integrator.
//
// Embedded explicit Runge-Kutta pair with 7 stages, fifth-order primary
// solution and fourth-order error estimate. Step size is controlled by the
// standard formula
//   dt_new = dt * clip(safety * err^{-1/5}, dt_shrink, dt_grow)
// where err is the weighted norm of (y5 - y4) divided by atol + rtol * |y|.
// Rejected steps retry with the reduced dt; the FSAL property (k7 of an
// accepted step equals k1 of the next) is exploited to save one RHS eval
// per accepted step.
//
// State requirements: arithmetic ops as for the fixed-step driver above,
// plus a scalar "norm" supplied by the caller as a callable
//   double norm(const State&)
// returning a non-negative magnitude. Defaults for double / std::complex /
// armadillo dense types are provided via state_norm() below; for custom
// state types either supply your own callable or overload state_norm().
// =====================================================================

struct AdaptiveOptions {
  double atol = 1e-8;
  double rtol = 1e-6;
  double dt_min = 1e-12;
  double dt_max = 0.0;  // 0 => unbounded (capped only by t1 - t)
  double safety = 0.9;
  double dt_grow = 5.0;
  double dt_shrink = 0.1;
  std::size_t max_rejected = 100;  // consecutive rejections before giving up
};

inline double state_norm(double x) noexcept { return std::abs(x); }
inline double state_norm(const std::complex<double>& z) noexcept { return std::abs(z); }

// Pulled in lazily via ADL for armadillo and user-defined types — if the
// caller's <armadillo> include is in scope, overload resolution will pick
// up arma::norm-based overloads defined alongside their state types.

struct DefaultStateNorm {
  template <typename State>
  double operator()(const State& s) const {
    using ::state_norm;
    return state_norm(s);
  }
};

namespace detail {

// Dormand-Prince 5(4) tableau (Dormand & Prince, J. Comp. Appl. Math. 6, 1980).
constexpr double kDpC2 = 1.0 / 5.0;
constexpr double kDpC3 = 3.0 / 10.0;
constexpr double kDpC4 = 4.0 / 5.0;
constexpr double kDpC5 = 8.0 / 9.0;

constexpr double kDpA21 = 1.0 / 5.0;
constexpr double kDpA31 = 3.0 / 40.0;
constexpr double kDpA32 = 9.0 / 40.0;
constexpr double kDpA41 = 44.0 / 45.0;
constexpr double kDpA42 = -56.0 / 15.0;
constexpr double kDpA43 = 32.0 / 9.0;
constexpr double kDpA51 = 19372.0 / 6561.0;
constexpr double kDpA52 = -25360.0 / 2187.0;
constexpr double kDpA53 = 64448.0 / 6561.0;
constexpr double kDpA54 = -212.0 / 729.0;
constexpr double kDpA61 = 9017.0 / 3168.0;
constexpr double kDpA62 = -355.0 / 33.0;
constexpr double kDpA63 = 46732.0 / 5247.0;
constexpr double kDpA64 = 49.0 / 176.0;
constexpr double kDpA65 = -5103.0 / 18656.0;
constexpr double kDpA71 = 35.0 / 384.0;
constexpr double kDpA72 = 0.0;
constexpr double kDpA73 = 500.0 / 1113.0;
constexpr double kDpA74 = 125.0 / 192.0;
constexpr double kDpA75 = -2187.0 / 6784.0;
constexpr double kDpA76 = 11.0 / 84.0;

// Fifth-order solution coefficients coincide with row 7 (FSAL).
constexpr double kDpB1 = 35.0 / 384.0;
constexpr double kDpB3 = 500.0 / 1113.0;
constexpr double kDpB4 = 125.0 / 192.0;
constexpr double kDpB5 = -2187.0 / 6784.0;
constexpr double kDpB6 = 11.0 / 84.0;

// Error coefficients e_i = b_i - b*_i (b* is the 4th-order embedded solution).
constexpr double kDpE1 = 71.0 / 57600.0;
constexpr double kDpE3 = -71.0 / 16695.0;
constexpr double kDpE4 = 71.0 / 1920.0;
constexpr double kDpE5 = -17253.0 / 339200.0;
constexpr double kDpE6 = 22.0 / 525.0;
constexpr double kDpE7 = -1.0 / 40.0;

}  // namespace detail

// One Dormand-Prince step. Takes k1 = f(t, state) as input (so the caller
// can reuse the previous accepted step's k7 — FSAL) and returns the
// candidate next state, k7 = f(t+dt, y_next), and the unweighted error
// estimate (y5 - y4).
template <typename System, typename State>
struct DormandPrinceStep {
  State y_next;
  State k7;
  State error;
};

template <typename System, typename State>
DormandPrinceStep<System, State> dormand_prince_step(const System& system, double t,
                                                     double dt, const State& y,
                                                     const State& k1) {
  using namespace detail;
  const State k2 = system(t + kDpC2 * dt, y + (dt * kDpA21) * k1);
  const State k3 = system(t + kDpC3 * dt, y + (dt * kDpA31) * k1 + (dt * kDpA32) * k2);
  const State k4 = system(t + kDpC4 * dt,
                          y + (dt * kDpA41) * k1 + (dt * kDpA42) * k2 + (dt * kDpA43) * k3);
  const State k5 = system(t + kDpC5 * dt, y + (dt * kDpA51) * k1 + (dt * kDpA52) * k2 +
                                              (dt * kDpA53) * k3 + (dt * kDpA54) * k4);
  const State k6 = system(t + dt, y + (dt * kDpA61) * k1 + (dt * kDpA62) * k2 +
                                      (dt * kDpA63) * k3 + (dt * kDpA64) * k4 +
                                      (dt * kDpA65) * k5);
  State y_next = y + (dt * kDpB1) * k1 + (dt * kDpB3) * k3 + (dt * kDpB4) * k4 +
                 (dt * kDpB5) * k5 + (dt * kDpB6) * k6;
  const State k7 = system(t + dt, y_next);
  State error = (dt * kDpE1) * k1 + (dt * kDpE3) * k3 + (dt * kDpE4) * k4 +
                (dt * kDpE5) * k5 + (dt * kDpE6) * k6 + (dt * kDpE7) * k7;
  return DormandPrinceStep<System, State>{std::move(y_next), std::move(k7),
                                          std::move(error)};
}

template <typename System, typename State, typename Norm, typename Callback>
State integrate_adaptive(const System& system, State state, double t0, double t1,
                         double dt_init, const AdaptiveOptions& opts, Norm norm,
                         Callback callback) {
  if (t1 <= t0) {
    callback(t0, state);
    return state;
  }
  if (dt_init <= 0.0) {
    throw std::invalid_argument("integrate_adaptive: dt_init must be positive");
  }
  if (opts.atol <= 0.0 && opts.rtol <= 0.0) {
    throw std::invalid_argument("integrate_adaptive: at least one of atol, rtol must be > 0");
  }

  const double dt_cap = (opts.dt_max > 0.0) ? opts.dt_max : (t1 - t0);
  double dt = std::min(dt_init, dt_cap);

  double t = t0;
  callback(t, state);
  State k1 = system(t, state);  // FSAL seed; refreshed after every accepted step

  std::size_t rejected = 0;
  while (t < t1) {
    if (t + dt > t1) dt = t1 - t;
    if (dt < opts.dt_min) {
      throw std::runtime_error("integrate_adaptive: step size underflowed dt_min");
    }

    auto step = dormand_prince_step(system, t, dt, state, k1);

    const double err_norm = norm(step.error);
    const double y_norm = norm(state);
    const double y_next_norm = norm(step.y_next);
    const double scale = opts.atol + opts.rtol * std::max(y_norm, y_next_norm);
    const double err = (scale > 0.0) ? (err_norm / scale) : 0.0;

    if (err <= 1.0) {
      // Accept.
      t += dt;
      state = std::move(step.y_next);
      k1 = std::move(step.k7);  // FSAL: k7 at (t, y) = k1 at (t+dt, y_next)
      callback(t, state);
      rejected = 0;

      double factor;
      if (err == 0.0) {
        factor = opts.dt_grow;
      } else {
        factor = opts.safety * std::pow(err, -0.2);
        factor = std::min(opts.dt_grow, std::max(opts.dt_shrink, factor));
      }
      dt = std::min(dt_cap, dt * factor);
    } else {
      // Reject. Shrink and retry without advancing t. Do NOT reuse k1 here —
      // it is still f(t, state), which is what the retry needs.
      ++rejected;
      if (rejected > opts.max_rejected) {
        throw std::runtime_error(
            "integrate_adaptive: too many consecutive step rejections");
      }
      double factor = opts.safety * std::pow(err, -0.2);
      factor = std::max(opts.dt_shrink, factor);
      dt = std::max(opts.dt_min, dt * factor);
    }
  }
  return state;
}

template <typename System, typename State, typename Norm>
State integrate_adaptive(const System& system, State state, double t0, double t1,
                         double dt_init, const AdaptiveOptions& opts, Norm norm) {
  return integrate_adaptive(system, std::move(state), t0, t1, dt_init, opts,
                            std::move(norm), NoopCallback{});
}

template <typename System, typename State>
State integrate_adaptive(const System& system, State state, double t0, double t1,
                         double dt_init, const AdaptiveOptions& opts) {
  return integrate_adaptive(system, std::move(state), t0, t1, dt_init, opts,
                            DefaultStateNorm{}, NoopCallback{});
}

template <typename System, typename State>
State integrate_adaptive(const System& system, State state, double t0, double t1,
                         double dt_init) {
  return integrate_adaptive(system, std::move(state), t0, t1, dt_init,
                            AdaptiveOptions{}, DefaultStateNorm{}, NoopCallback{});
}
