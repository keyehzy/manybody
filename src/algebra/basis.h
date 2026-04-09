#pragma once

#include <cmath>
#include <vector>

#include "algebra/monomial.h"
#include "algebra/operator.h"
#include "algebra/statistics.h"
#include "utils/index.h"
#include "utils/indexed_hash_set.h"

template <typename MonomialType>
struct BasisImpl {
  using operator_type = typename MonomialType::operator_type;
  using scalar_type = typename MonomialType::scalar_type;
  using value_type = typename scalar_type::value_type;
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

  static value_type state_normalization(const key_type& state) noexcept {
    if constexpr (AlgebraTraits<statistics>::pauli_exclusion) {
      return value_type{1.0};
    } else {
      value_type normalization_squared{1.0};
      size_t multiplicity{0};
      operator_type previous{};

      for (const auto& op : state) {
        if (multiplicity == 0 || !(op == previous)) {
          multiplicity = 1;
          previous = op;
          continue;
        }

        ++multiplicity;
        normalization_squared /= static_cast<value_type>(multiplicity);
      }

      return std::sqrt(normalization_squared);
    }
  }

  value_type state_normalization(size_t index) const noexcept {
    return state_normalization(set[index]);
  }

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
