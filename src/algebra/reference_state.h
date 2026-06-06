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

  static constexpr complex_type contract_plain_then_plain(FermionOperator,
                                                          FermionOperator) noexcept {
    return complex_type{0.0, 0.0};
  }

  static constexpr complex_type contract_dagger_then_dagger(FermionOperator,
                                                            FermionOperator) noexcept {
    return complex_type{0.0, 0.0};
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

  complex_type contract_plain_then_plain(FermionOperator, FermionOperator) const noexcept {
    return complex_type{0.0, 0.0};
  }

  complex_type contract_dagger_then_dagger(FermionOperator, FermionOperator) const noexcept {
    return complex_type{0.0, 0.0};
  }

  complex_type double_occupancy_cumulant(std::size_t orbital) const noexcept {
    const complex_type n_up = density(FermionOperator::Spin::Up, orbital);
    const complex_type n_dn = density(FermionOperator::Spin::Down, orbital);
    return D - n_up * n_dn;
  }
};

// Quasi-free reference parameterised by a fully general one-body density matrix
// rho_{a,b} = <c+_a c_b>, where the combined index a = spin * n_orbital + orbital
// runs over both spin sectors. Storage is row-major in a flat vector of length
// dim()^2 = (2 n_orbital)^2; no Armadillo dependency.
//
// Suitable as a reference for CDW/SDW or inhomogeneous states whose one-body
// density matrix is not diagonal in the chosen single-particle basis. Anomalous
// expectations (<c c>, <c+ c+>) are zero, so this does not describe BCS-like
// references; the on-site double-occupancy cumulant is also zero by default
// (a general 4-point cumulant tensor would be the natural extension).
struct GeneralFermionReference {
  using complex_type = std::complex<double>;

  std::size_t n_orbital{0};
  std::vector<complex_type> rho{};

  GeneralFermionReference() = default;

  explicit GeneralFermionReference(std::size_t n_orb)
      : n_orbital(n_orb), rho(4 * n_orb * n_orb, complex_type{0.0, 0.0}) {}

  GeneralFermionReference(std::size_t n_orb,
                          const std::vector<std::array<complex_type, 2>>& n_diagonal)
      : n_orbital(n_orb), rho(4 * n_orb * n_orb, complex_type{0.0, 0.0}) {
    for (std::size_t k = 0; k < n_orb; ++k) {
      for (std::size_t s = 0; s < 2; ++s) {
        const std::size_t idx = s * n_orb + k;
        rho_at(idx, idx) = n_diagonal[k][s];
      }
    }
  }

  std::size_t dim() const noexcept { return 2 * n_orbital; }

  std::size_t flat_index(FermionOperator::Spin spin, std::size_t orbital) const noexcept {
    return static_cast<std::size_t>(spin) * n_orbital + orbital;
  }

  complex_type& rho_at(std::size_t row, std::size_t col) noexcept { return rho[row * dim() + col]; }
  complex_type rho_at(std::size_t row, std::size_t col) const noexcept {
    return rho[row * dim() + col];
  }

  complex_type density(FermionOperator::Spin spin, std::size_t k) const noexcept {
    const std::size_t idx = flat_index(spin, k);
    return rho_at(idx, idx);
  }

  complex_type contract_dagger_then_plain(FermionOperator a, FermionOperator b) const noexcept {
    return rho_at(flat_index(a.spin(), a.value()), flat_index(b.spin(), b.value()));
  }

  complex_type contract_plain_then_dagger(FermionOperator a, FermionOperator b) const noexcept {
    const std::size_t i_a = flat_index(a.spin(), a.value());
    const std::size_t i_b = flat_index(b.spin(), b.value());
    const complex_type kronecker = (i_a == i_b) ? complex_type{1.0, 0.0} : complex_type{0.0, 0.0};
    return kronecker - rho_at(i_b, i_a);
  }

  complex_type contract_plain_then_plain(FermionOperator, FermionOperator) const noexcept {
    return complex_type{0.0, 0.0};
  }

  complex_type contract_dagger_then_dagger(FermionOperator, FermionOperator) const noexcept {
    return complex_type{0.0, 0.0};
  }

  complex_type double_occupancy_cumulant(std::size_t /*orbital*/) const noexcept {
    return complex_type{0.0, 0.0};
  }
};

// Quasi-free reference parameterised by the full generalised one-body density
// matrix Gamma in Nambu/BdG form. Allows non-zero anomalous (pair) expectations
// <c_a c_b> = kappa_{a,b}, suitable as a reference for BCS-like superconducting
// states or for flowing the attractive Hubbard model.
//
// Nambu spinor convention with single-particle dimension N = 2 * n_orbital
// (covering both spins) and Nambu dimension 2N = 4 * n_orbital:
//   Psi_I = c_I        for 0 <= I < N
//   Psi_I = c+_{I-N}   for N <= I < 2N
// The stored matrix is Gamma_{IJ} = <Psi+_I Psi_J>, row-major, shape (2N)^2.
// Block structure:
//   Gamma = [ rho        <c+ c+> ]
//           [ <c c>=kappa  1 - rho^T ]
// Physical states satisfy Gamma = Gamma+ (Hermitian), kappa antisymmetric.
// No validation is performed; trust-the-user. The on-site D cumulant is zero
// here (BCS-mean-field-level pair physics is already captured by Gaussian
// Wick contractions through the kappa block).
struct NambuFermionReference {
  using complex_type = std::complex<double>;

  std::size_t n_orbital{0};
  std::vector<complex_type> gamma{};

  NambuFermionReference() = default;

  explicit NambuFermionReference(std::size_t n_orb)
      : n_orbital(n_orb), gamma(16 * n_orb * n_orb, complex_type{0.0, 0.0}) {}

  explicit NambuFermionReference(const GeneralFermionReference& normal)
      : n_orbital(normal.n_orbital),
        gamma(16 * normal.n_orbital * normal.n_orbital, complex_type{0.0, 0.0}) {
    const std::size_t N = single_particle_dim();
    for (std::size_t i = 0; i < N; ++i) {
      for (std::size_t j = 0; j < N; ++j) {
        gamma_at(i, j) = normal.rho_at(i, j);
        const complex_type kronecker = (i == j) ? complex_type{1.0, 0.0} : complex_type{0.0, 0.0};
        gamma_at(N + i, N + j) = kronecker - normal.rho_at(j, i);
      }
    }
  }

  std::size_t single_particle_dim() const noexcept { return 2 * n_orbital; }
  std::size_t nambu_dim() const noexcept { return 4 * n_orbital; }

  std::size_t flat_index(FermionOperator::Spin spin, std::size_t orbital) const noexcept {
    return static_cast<std::size_t>(spin) * n_orbital + orbital;
  }

  std::size_t particle_index(FermionOperator::Spin spin, std::size_t orbital) const noexcept {
    return flat_index(spin, orbital);
  }

  std::size_t hole_index(FermionOperator::Spin spin, std::size_t orbital) const noexcept {
    return single_particle_dim() + flat_index(spin, orbital);
  }

  complex_type& gamma_at(std::size_t I, std::size_t J) noexcept {
    return gamma[I * nambu_dim() + J];
  }
  complex_type gamma_at(std::size_t I, std::size_t J) const noexcept {
    return gamma[I * nambu_dim() + J];
  }

  // Sets <c+_a c_a> = n consistently in both the rho block and the (1 - rho^T) block.
  void set_density(FermionOperator::Spin spin, std::size_t orbital, complex_type n) noexcept {
    const std::size_t a = flat_index(spin, orbital);
    const std::size_t N = single_particle_dim();
    gamma_at(a, a) = n;
    gamma_at(N + a, N + a) = complex_type{1.0, 0.0} - n;
  }

  // Sets <c_a c_b> = alpha with full antisymmetric and Hermitian consistency:
  //   <c_a c_b> = alpha, <c_b c_a> = -alpha
  //   <c+_a c+_b> = -alpha*, <c+_b c+_a> = alpha*
  void set_pair_amplitude(FermionOperator::Spin sa, std::size_t ka, FermionOperator::Spin sb,
                          std::size_t kb, complex_type alpha) noexcept {
    const std::size_t a = flat_index(sa, ka);
    const std::size_t b = flat_index(sb, kb);
    const std::size_t N = single_particle_dim();
    gamma_at(N + a, b) = alpha;
    gamma_at(N + b, a) = -alpha;
    gamma_at(a, N + b) = -std::conj(alpha);
    gamma_at(b, N + a) = std::conj(alpha);
  }

  complex_type density(FermionOperator::Spin spin, std::size_t k) const noexcept {
    const std::size_t a = flat_index(spin, k);
    return gamma_at(a, a);
  }

  complex_type contract_dagger_then_plain(FermionOperator a, FermionOperator b) const noexcept {
    return gamma_at(particle_index(a.spin(), a.value()), particle_index(b.spin(), b.value()));
  }

  complex_type contract_plain_then_dagger(FermionOperator a, FermionOperator b) const noexcept {
    return gamma_at(hole_index(a.spin(), a.value()), hole_index(b.spin(), b.value()));
  }

  complex_type contract_plain_then_plain(FermionOperator a, FermionOperator b) const noexcept {
    return gamma_at(hole_index(a.spin(), a.value()), particle_index(b.spin(), b.value()));
  }

  complex_type contract_dagger_then_dagger(FermionOperator a, FermionOperator b) const noexcept {
    return gamma_at(particle_index(a.spin(), a.value()), hole_index(b.spin(), b.value()));
  }

  complex_type double_occupancy_cumulant(std::size_t /*orbital*/) const noexcept {
    return complex_type{0.0, 0.0};
  }
};
