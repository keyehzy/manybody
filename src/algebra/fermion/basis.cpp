#include "algebra/fermion/basis.h"

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

constexpr uint64_t compute_basis_size(uint64_t orbitals, uint64_t particles) noexcept {
  return choose(2 * orbitals, particles);
}

constexpr uint64_t compute_basis_size_spin(uint64_t orbitals, uint64_t up, uint64_t down) noexcept {
  return choose(orbitals, up) * choose(orbitals, down);
}

Basis Basis::with_fixed_particle_number(size_t orbitals, size_t particles) {
  Basis basis;
  basis.orbitals = orbitals;
  basis.particles = particles;
  assert(orbitals <= Operator::max_index());
  assert(particles <= 2 * orbitals);

  size_t basis_size = compute_basis_size(orbitals, particles);
  std::vector<key_type> acc;
  acc.reserve(basis_size);
  basis.generate_restrict_combinations({}, 0, acc);
  assert(acc.size() == basis_size);
  basis.set = IndexedHashSet(std::move(acc));
  return basis;
}

Basis Basis::with_fixed_particle_number_and_spin(size_t orbitals, size_t particles,
                                                 int spin_projection) {
  Basis basis;
  basis.orbitals = orbitals;
  basis.particles = particles;
  assert(orbitals <= Operator::max_index());
  assert(particles <= 2 * orbitals);

  const int64_t particles_signed = static_cast<int64_t>(particles);
  const int64_t spin = static_cast<int64_t>(spin_projection);
  assert(std::abs(spin) <= particles_signed);
  assert((particles_signed + spin) % 2 == 0);

  const int64_t up_signed = (particles_signed + spin) / 2;
  const int64_t down_signed = particles_signed - up_signed;
  assert(up_signed >= 0 && down_signed >= 0);
  const size_t up = static_cast<size_t>(up_signed);
  const size_t down = static_cast<size_t>(down_signed);
  assert(up <= orbitals);
  assert(down <= orbitals);

  size_t basis_size = compute_basis_size_spin(orbitals, up, down);
  std::vector<key_type> acc;
  acc.reserve(basis_size);
  basis.generate_restrict_combinations_with_spin({}, 0, up, down, acc);
  assert(acc.size() == basis_size);
  basis.set = IndexedHashSet(std::move(acc));
  return basis;
}

Basis Basis::with_fixed_particle_number_spin_momentum(size_t orbitals, size_t particles,
                                                      int spin_projection, const Index& index,
                                                      const Index::container_type& momentum) {
  Basis basis;
  basis.orbitals = orbitals;
  basis.particles = particles;

  assert(orbitals <= Operator::max_index());
  assert(particles <= 2 * orbitals);
  assert(index.size() == orbitals);

  const auto& dims = index.dimensions();
  assert(momentum.size() == dims.size());
  for (size_t d = 0; d < dims.size(); ++d) {
    assert(dims[d] > 0);
    assert(momentum[d] < dims[d]);
  }

  const int64_t particles_signed = static_cast<int64_t>(particles);
  const int64_t spin = static_cast<int64_t>(spin_projection);
  assert(std::abs(spin) <= particles_signed);
  assert((particles_signed + spin) % 2 == 0);

  const int64_t up_signed = (particles_signed + spin) / 2;
  const int64_t down_signed = particles_signed - up_signed;
  assert(up_signed >= 0 && down_signed >= 0);

  const size_t up = static_cast<size_t>(up_signed);
  const size_t down = static_cast<size_t>(down_signed);
  assert(up <= orbitals);
  assert(down <= orbitals);

  // Precompute per-orbital momentum coordinates once
  std::vector<Index::container_type> orbital_coords;
  orbital_coords.reserve(orbitals);
  for (size_t i = 0; i < orbitals; ++i) orbital_coords.push_back(index(i));

  // Upper bound reserve (sector will be <= full spin-resolved size)
  size_t max_size = compute_basis_size_spin(orbitals, up, down);

  std::vector<key_type> acc;
  acc.reserve(max_size);

  key_type current;

  Index::container_type current_momentum(dims.size(), 0);

  basis.generate_restrict_combinations_with_spin_momentum(current, 0, up, down, current_momentum,
                                                          orbital_coords, index, momentum, acc);

  basis.set = IndexedHashSet(std::move(acc));
  return basis;
}

Basis Basis::with_all_particle_number(size_t orbitals, size_t particles) {
  Basis basis;
  basis.orbitals = orbitals;
  basis.particles = particles;
  assert(orbitals <= Operator::max_index());
  assert(particles <= 2 * orbitals);

  size_t basis_size = 0;
  for (size_t i = 1; i <= particles; ++i) {
    basis_size += compute_basis_size(orbitals, i);
  }
  std::vector<key_type> acc;
  acc.reserve(basis_size);
  basis.generate_all_combinations({}, 0, acc);
  assert(acc.size() == basis_size);
  std::sort(acc.begin(), acc.end(),
            [](const auto& a, const auto& b) { return a.size() < b.size(); });
  basis.set = IndexedHashSet(std::move(acc));
  return basis;
}

void Basis::generate_all_combinations(key_type current, size_t first_orbital,
                                      std::vector<key_type>& acc) const {
  if (current.size() > 0 && current.size() <= particles) {
    std::sort(current.begin(), current.end(), [](const auto& a, const auto& b) { return a < b; });
    acc.emplace_back(current);
  }
  for (int spin_index = 0; spin_index < 2; ++spin_index) {
    for (size_t i = first_orbital; i < orbitals; i++) {
      Operator::Spin spin = static_cast<Operator::Spin>(spin_index);
      bool should_iterate =
          current.empty() || (current.back().value() < i ||
                              (current.back().value() == i && spin > current.back().spin()));
      if (should_iterate) {
        current.push_back(Operator::creation(spin, i));
        generate_all_combinations(current, i, acc);
        current.pop_back();
      }
    }
  }
}

void Basis::generate_restrict_combinations(key_type current, size_t first_orbital,
                                           std::vector<key_type>& acc) const {
  if (current.size() == particles) {
    std::sort(current.begin(), current.end(), [](const auto& a, const auto& b) { return a < b; });
    acc.emplace_back(current);
    return;
  }
  for (int spin_index = 0; spin_index < 2; ++spin_index) {
    for (size_t i = first_orbital; i < orbitals; i++) {
      Operator::Spin spin = static_cast<Operator::Spin>(spin_index);
      bool should_iterate =
          current.empty() || (current.back().value() < i ||
                              (current.back().value() == i && spin > current.back().spin()));
      if (should_iterate) {
        current.push_back(Operator::creation(spin, i));
        generate_restrict_combinations(current, i, acc);
        current.pop_back();
      }
    }
  }
}

void Basis::generate_restrict_combinations_with_spin(key_type current, size_t first_orbital,
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
      Operator::Spin spin = static_cast<Operator::Spin>(spin_index);
      if ((spin == Operator::Spin::Up && up_left == 0) ||
          (spin == Operator::Spin::Down && down_left == 0)) {
        continue;
      }
      bool should_iterate =
          current.empty() || (current.back().value() < i ||
                              (current.back().value() == i && spin > current.back().spin()));
      if (should_iterate) {
        current.push_back(Operator::creation(spin, i));
        generate_restrict_combinations_with_spin(
            current, i, up_left - (spin == Operator::Spin::Up ? 1 : 0),
            down_left - (spin == Operator::Spin::Down ? 1 : 0), acc);
        current.pop_back();
      }
    }
  }
}

namespace {
inline void add_momentum(Index::container_type& total, const Index::container_type& add,
                         const std::vector<size_t>& dims) {
  for (size_t d = 0; d < dims.size(); ++d) {
    total[d] += add[d];
    total[d] %= dims[d];
  }
}

inline void sub_momentum(Index::container_type& total, const Index::container_type& sub,
                         const std::vector<size_t>& dims) {
  for (size_t d = 0; d < dims.size(); ++d) {
    total[d] = (total[d] + dims[d] - (sub[d] % dims[d])) % dims[d];
  }
}
}  // namespace

void Basis::generate_restrict_combinations_with_spin_momentum(
    key_type& current, size_t first_orbital, size_t up_left, size_t down_left,
    Index::container_type& current_momentum,
    const std::vector<Index::container_type>& orbital_coords, const Index& index,
    const Index::container_type& target_momentum, std::vector<key_type>& acc) const {
  if (up_left == 0 && down_left == 0) {
    if (current_momentum == target_momentum) {
      key_type sorted = current;
      std::sort(sorted.begin(), sorted.end(), [](const auto& a, const auto& b) { return a < b; });
      acc.emplace_back(std::move(sorted));
    }
    return;
  }

  if (current.size() >= particles) {
    return;
  }

  const auto& dims = index.dimensions();

  for (int spin_index = 0; spin_index < 2; ++spin_index) {
    for (size_t i = first_orbital; i < orbitals; ++i) {
      Operator::Spin spin = static_cast<Operator::Spin>(spin_index);

      if ((spin == Operator::Spin::Up && up_left == 0) ||
          (spin == Operator::Spin::Down && down_left == 0)) {
        continue;
      }

      bool should_iterate =
          current.empty() || (current.back().value() < i ||
                              (current.back().value() == i && spin > current.back().spin()));

      if (should_iterate) {
        // Choose
        current.push_back(Operator::creation(spin, i));
        add_momentum(current_momentum, orbital_coords[i], dims);

        // Recurse
        generate_restrict_combinations_with_spin_momentum(
            current, i, up_left - (spin == Operator::Spin::Up ? 1 : 0),
            down_left - (spin == Operator::Spin::Down ? 1 : 0), current_momentum, orbital_coords,
            index, target_momentum, acc);

        // Undo
        sub_momentum(current_momentum, orbital_coords[i], dims);
        current.pop_back();
      }
    }
  }
}
