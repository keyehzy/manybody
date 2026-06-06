#pragma once

#include <array>
#include <complex>
#include <cstddef>
#include <vector>

#include "algebra/operator.h"

struct VacuumFermionReference {
  using complex_type = std::complex<double>;

  static constexpr complex_type density(FermionOperator::Spin, std::size_t) noexcept {
    return complex_type{0.0, 0.0};
  }

  static constexpr complex_type contract_dagger_then_plain(FermionOperator,
                                                           FermionOperator) noexcept {
    return complex_type{0.0, 0.0};
  }

  static complex_type contract_plain_then_dagger(FermionOperator a, FermionOperator b) noexcept {
    return (a.spin() == b.spin() && a.value() == b.value()) ? complex_type{1.0, 0.0}
                                                            : complex_type{0.0, 0.0};
  }

  static constexpr complex_type double_occupancy_cumulant(std::size_t) noexcept {
    return complex_type{0.0, 0.0};
  }
};

struct DiagonalFermionReference {
  using complex_type = std::complex<double>;

  std::vector<std::array<complex_type, 2>> n{};
  complex_type D{0.0, 0.0};

  DiagonalFermionReference() = default;
  DiagonalFermionReference(std::size_t n_sites, complex_type n_default, complex_type D_value)
      : n(n_sites, std::array<complex_type, 2>{n_default, n_default}), D(D_value) {}

  complex_type density(FermionOperator::Spin spin, std::size_t k) const noexcept {
    return n[k][static_cast<std::size_t>(spin)];
  }

  complex_type contract_dagger_then_plain(FermionOperator a, FermionOperator b) const noexcept {
    if (a.spin() != b.spin() || a.value() != b.value()) {
      return complex_type{0.0, 0.0};
    }
    return density(a.spin(), a.value());
  }

  complex_type contract_plain_then_dagger(FermionOperator a, FermionOperator b) const noexcept {
    if (a.spin() != b.spin() || a.value() != b.value()) {
      return complex_type{0.0, 0.0};
    }
    return complex_type{1.0, 0.0} - density(a.spin(), a.value());
  }

  complex_type double_occupancy_cumulant(std::size_t orbital) const noexcept {
    const complex_type n_up = density(FermionOperator::Spin::Up, orbital);
    const complex_type n_dn = density(FermionOperator::Spin::Down, orbital);
    return D - n_up * n_dn;
  }
};
