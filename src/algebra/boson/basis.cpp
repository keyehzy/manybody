#include "algebra/boson/basis.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <vector>

constexpr uint64_t choose(uint64_t n, uint64_t m) noexcept {
  if (m > n) return 0;
  if (m == 0 || m == n) return 1;

  if (m > n - m) m = n - m;

  uint64_t result = 1;
  for (uint64_t i = 0; i < m; ++i) {
    result *= (n - i);
    result /= (i + 1);
  }

  return result;
}

// Stars-and-bars: C(modes + particles - 1, particles)
constexpr uint64_t compute_boson_basis_size(uint64_t modes, uint64_t particles) noexcept {
  if (modes == 0) return (particles == 0) ? 1 : 0;
  return choose(modes + particles - 1, particles);
}

constexpr uint64_t compute_boson_basis_size_spin(uint64_t orbitals, uint64_t up,
                                                 uint64_t down) noexcept {
  return compute_boson_basis_size(orbitals, up) * compute_boson_basis_size(orbitals, down);
}

BosonBasis BosonBasis::with_fixed_particle_number(size_t orbitals, size_t particles) {
  BosonBasis basis;
  basis.orbitals = orbitals;
  basis.particles = particles;
  assert(orbitals <= BosonOperator::max_index());
  assert(particles <= BosonMonomial::container_type::max_size());

  size_t basis_size = compute_boson_basis_size(2 * orbitals, particles);
  std::vector<key_type> acc;
  acc.reserve(basis_size);
  basis.generate_restrict_combinations({}, 0, acc);
  assert(acc.size() == basis_size);
  basis.set = IndexedHashSet(std::move(acc));
  return basis;
}

BosonBasis BosonBasis::with_fixed_particle_number_and_spin(size_t orbitals, size_t particles,
                                                           int spin_projection) {
  BosonBasis basis;
  basis.orbitals = orbitals;
  basis.particles = particles;
  assert(orbitals <= BosonOperator::max_index());
  assert(particles <= BosonMonomial::container_type::max_size());

  const int64_t particles_signed = static_cast<int64_t>(particles);
  const int64_t spin = static_cast<int64_t>(spin_projection);
  assert(std::abs(spin) <= particles_signed);
  assert((particles_signed + spin) % 2 == 0);

  const int64_t up_signed = (particles_signed + spin) / 2;
  const int64_t down_signed = particles_signed - up_signed;
  assert(up_signed >= 0 && down_signed >= 0);
  const size_t up = static_cast<size_t>(up_signed);
  const size_t down = static_cast<size_t>(down_signed);

  size_t basis_size = compute_boson_basis_size_spin(orbitals, up, down);
  std::vector<key_type> acc;
  acc.reserve(basis_size);
  basis.generate_restrict_combinations_with_spin({}, 0, up, down, acc);
  assert(acc.size() == basis_size);
  basis.set = IndexedHashSet(std::move(acc));
  return basis;
}

void BosonBasis::generate_restrict_combinations(key_type current, size_t first_orbital,
                                                std::vector<key_type>& acc) const {
  if (current.size() == particles) {
    std::sort(current.begin(), current.end(), [](const auto& a, const auto& b) { return a < b; });
    acc.emplace_back(current);
    return;
  }
  for (int spin_index = 0; spin_index < 2; ++spin_index) {
    for (size_t i = first_orbital; i < orbitals; i++) {
      BosonOperator::Spin spin = static_cast<BosonOperator::Spin>(spin_index);
      // Non-strict ordering: >= instead of > allows repeated (spin, orbital)
      bool should_iterate =
          current.empty() || (current.back().value() < i ||
                              (current.back().value() == i && spin >= current.back().spin()));
      if (should_iterate) {
        current.push_back(BosonOperator::creation(spin, i));
        generate_restrict_combinations(current, i, acc);
        current.pop_back();
      }
    }
  }
}

void BosonBasis::generate_restrict_combinations_with_spin(key_type current, size_t first_orbital,
                                                          size_t up_left, size_t down_left,
                                                          std::vector<key_type>& acc) const {
  if (up_left == 0 && down_left == 0) {
    std::sort(current.begin(), current.end(), [](const auto& a, const auto& b) { return a < b; });
    acc.emplace_back(current);
    return;
  }
  if (current.size() >= particles) {
    return;
  }
  for (int spin_index = 0; spin_index < 2; ++spin_index) {
    for (size_t i = first_orbital; i < orbitals; i++) {
      BosonOperator::Spin spin = static_cast<BosonOperator::Spin>(spin_index);
      if ((spin == BosonOperator::Spin::Up && up_left == 0) ||
          (spin == BosonOperator::Spin::Down && down_left == 0)) {
        continue;
      }
      // Non-strict ordering: >= instead of > allows repeated (spin, orbital)
      bool should_iterate =
          current.empty() || (current.back().value() < i ||
                              (current.back().value() == i && spin >= current.back().spin()));
      if (should_iterate) {
        current.push_back(BosonOperator::creation(spin, i));
        generate_restrict_combinations_with_spin(
            current, i, up_left - (spin == BosonOperator::Spin::Up ? 1 : 0),
            down_left - (spin == BosonOperator::Spin::Down ? 1 : 0), acc);
        current.pop_back();
      }
    }
  }
}
