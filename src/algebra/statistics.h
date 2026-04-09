#pragma once

enum class Statistics { Fermion, Boson };

template <Statistics S>
struct AlgebraTraits;

template <>
struct AlgebraTraits<Statistics::Fermion> {
  static constexpr int swap_sign = -1;
  static constexpr bool pauli_exclusion = true;
  static constexpr int contraction_sign = -1;
  static constexpr const char* label = "c";
};

template <>
struct AlgebraTraits<Statistics::Boson> {
  static constexpr int swap_sign = +1;
  static constexpr bool pauli_exclusion = false;
  static constexpr int contraction_sign = +1;
  static constexpr const char* label = "b";
};
