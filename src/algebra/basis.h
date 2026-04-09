#pragma once

#include <vector>

#include "algebra/monomial.h"
#include "algebra/operator.h"
#include "algebra/statistics.h"
#include "utils/index.h"
#include "utils/indexed_hash_set.h"

template <typename MonomialType>
struct BasisImpl {
  using operator_type = typename MonomialType::operator_type;
  using key_type = typename MonomialType::container_type;
  using set_type = IndexedHashSet<key_type>;

  static constexpr Statistics statistics = operator_type::statistics;

  BasisImpl() = default;

  static BasisImpl with_fixed_particle_number(size_t orbitals, size_t particles);
  static BasisImpl with_fixed_particle_number_and_spin(size_t orbitals, size_t particles,
                                                       int spin_projection);
  static BasisImpl with_fixed_particle_number_spin_momentum(size_t orbitals, size_t particles,
                                                            int spin_projection, const Index& index,
                                                            const Index::container_type& momentum);
  static BasisImpl with_all_particle_number(size_t orbitals, size_t particles);

  void generate_all_combinations(key_type current, size_t first_orbital,
                                 std::vector<key_type>&) const;

  void generate_restrict_combinations(key_type current, size_t first_orbital,
                                      std::vector<key_type>&) const;
  void generate_restrict_combinations_with_spin(key_type current, size_t first_orbital,
                                                size_t up_left, size_t down_left,
                                                std::vector<key_type>& acc) const;
  void generate_restrict_combinations_with_spin_momentum(
      key_type& current, size_t first_orbital, size_t up_left, size_t down_left,
      Index::container_type& current_momentum,
      const std::vector<Index::container_type>& orbital_coords, const Index& index,
      const Index::container_type& target_momentum, std::vector<key_type>& acc) const;

  set_type set{};
  size_t orbitals{0};
  size_t particles{0};
};
