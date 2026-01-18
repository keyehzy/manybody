#pragma once

#include <algorithm>
#include <cmath>
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
