#pragma once

#include "indexed_hash_set.h"
#include "term.h"

struct Basis {
  using key_type = Term::container_type;
  using set_type = IndexedHashSet<key_type>;

  enum class Strategy {
    All,
    Restrict,
  };

  Basis() = default;

  Basis(size_t orbitals, size_t particles, Strategy strategy = Strategy::Restrict);

  void generate_all_combinations(key_type current, size_t first_orbital,
                                 std::vector<key_type>&) const;

  void generate_restrict_combinations(key_type current, size_t first_orbital,
                                      std::vector<key_type>&) const;

  set_type set;
  size_t orbitals;
  size_t particles;
};
