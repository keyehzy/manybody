#pragma once

#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

#include "operator.h"

class DynamicIndex {
 public:
  using size_type = std::size_t;
  using container_type = std::vector<size_type>;

  explicit DynamicIndex(container_type dimensions) : dimensions_(std::move(dimensions)) {
    if (total_size() > Operator::max_index()) {
      throw std::out_of_range("Index size exceeds maximum orbitals.");
    }
  }

  [[nodiscard]] size_type to_orbital(const container_type& coordinates) const {
    if (coordinates.size() != dimensions_.size()) {
      throw std::out_of_range("Invalid number of coordinates.");
    }

    size_type orbital = 0;
    size_type multiplier = 1;
    for (size_type i = 0; i < dimensions_.size(); ++i) {
      if (coordinates[i] >= dimensions_[i]) {
        throw std::out_of_range("Coordinates out of bounds.");
      }
      orbital += coordinates[i] * multiplier;
      multiplier *= dimensions_[i];
    }
    return orbital;
  }

  [[nodiscard]] container_type from_orbital(size_type orbital) const {
    if (orbital >= total_size()) {
      throw std::out_of_range("Orbital index out of bounds.");
    }

    container_type coordinates(dimensions_.size());
    for (size_type i = 0; i < dimensions_.size(); ++i) {
      coordinates[i] = orbital % dimensions_[i];
      orbital /= dimensions_[i];
    }
    return coordinates;
  }

  [[nodiscard]] size_type value_at(size_type orbital, size_type dimension) const {
    if (dimension >= dimensions_.size()) {
      throw std::out_of_range("Dimension is out of bounds.");
    }
    return from_orbital(orbital)[dimension];
  }

  const container_type& dimensions() const noexcept { return dimensions_; }

  size_type dimension(size_type index) const {
    if (index >= dimensions_.size()) {
      throw std::out_of_range("Dimension is out of bounds.");
    }
    return dimensions_[index];
  }

  [[nodiscard]] size_type size() const { return total_size(); }

 private:
  container_type dimensions_;

  size_type total_size() const {
    size_type size = 1;
    const size_type max_orbitals = Operator::max_index();
    for (size_type dim : dimensions_) {
      if (dim == 0) {
        return 0;
      }
      if (size > max_orbitals / dim) {
        return max_orbitals + 1;
      }
      size *= dim;
    }
    return size;
  }
};
