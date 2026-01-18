#pragma once

#include "algebra/term.h"
#include "utils/index.h"
#include "utils/indexed_hash_set.h"

struct Basis {
  using key_type = Term::container_type;
  using set_type = IndexedHashSet<key_type>;

  Basis() = default;

  static Basis with_fixed_particle_number(size_t orbitals, size_t particles);
  static Basis with_fixed_particle_number_and_spin(size_t orbitals, size_t particles,
                                                   int spin_projection);
  static Basis with_fixed_particle_number_spin_momentum(size_t orbitals, size_t particles,
                                                        int spin_projection, const Index& index,
                                                        const Index::container_type& momentum);
  static Basis with_all_particle_number(size_t orbitals, size_t particles);

  void generate_all_combinations(key_type current, size_t first_orbital,
                                 std::vector<key_type>&) const;

  void generate_restrict_combinations(key_type current, size_t first_orbital,
                                      std::vector<key_type>&) const;
  void generate_restrict_combinations_with_spin(key_type current, size_t first_orbital,
                                                size_t up_left, size_t down_left,
                                                std::vector<key_type>& acc) const;

  set_type set{};
  size_t orbitals{0};
  size_t particles{0};
};
