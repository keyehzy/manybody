#pragma once

#include <vector>

#include "algebra/boson/term.h"
#include "utils/indexed_hash_set.h"

struct BosonBasis {
  using key_type = BosonMonomial::container_type;
  using set_type = IndexedHashSet<key_type>;

  BosonBasis() = default;

  static BosonBasis with_fixed_particle_number(size_t orbitals, size_t particles);
  static BosonBasis with_fixed_particle_number_and_spin(size_t orbitals, size_t particles,
                                                        int spin_projection);

  void generate_restrict_combinations(key_type current, size_t first_orbital,
                                      std::vector<key_type>& acc) const;

  void generate_restrict_combinations_with_spin(key_type current, size_t first_orbital,
                                                size_t up_left, size_t down_left,
                                                std::vector<key_type>& acc) const;

  set_type set{};
  size_t orbitals{0};
  size_t particles{0};
};
