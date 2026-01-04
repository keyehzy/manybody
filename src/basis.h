#pragma once

#include "indexed_hash_set.h"
#include "term.h"

struct Basis {
  using key_type = Term::container_type;
  using set_type = IndexedHashSet<key_type>;

  Basis() = default;

  static Basis with_fixed_particle_number(size_t orbitals, size_t particles);
  static Basis with_all_particle_number(size_t orbitals, size_t particles);

  void generate_all_combinations(key_type current, size_t first_orbital,
                                 std::vector<key_type>&) const;

  void generate_restrict_combinations(key_type current, size_t first_orbital,
                                      std::vector<key_type>&) const;

  set_type set;
  size_t orbitals;
  size_t particles;

 private:
  enum class Strategy {
    All,
    Restrict,
  };

  Basis(size_t orbitals, size_t particles, Strategy strategy);
};
