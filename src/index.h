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
  using offset_type = std::vector<int>;

  explicit DynamicIndex(container_type dimensions) : dimensions_(std::move(dimensions)) {}

  explicit DynamicIndex(std::initializer_list<size_type> dimensions)
      : dimensions_(std::move(dimensions)) {}

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

  [[nodiscard]] size_type operator()(const container_type& coordinates) const {
    return to_orbital(coordinates);
  }

  [[nodiscard]] size_type operator()(std::initializer_list<size_type> coordinates) const {
    return to_orbital(container_type(coordinates));
  }

  [[nodiscard]] size_type operator()(const container_type& coordinates,
                                     const offset_type& offsets) const {
    return to_orbital(wrap(coordinates, offsets));
  }

  [[nodiscard]] size_type operator()(std::initializer_list<size_type> coordinates,
                                     std::initializer_list<int> offsets) const {
    return to_orbital(wrap(container_type(coordinates), offset_type(offsets)));
  }

  [[nodiscard]] container_type wrap(const container_type& coordinates,
                                    const offset_type& offsets) const {
    if (coordinates.size() != dimensions_.size()) {
      throw std::out_of_range("Invalid number of coordinates.");
    }
    if (offsets.size() != dimensions_.size()) {
      throw std::out_of_range("Invalid number of offsets.");
    }

    container_type wrapped(dimensions_.size());
    for (size_type i = 0; i < dimensions_.size(); ++i) {
      const int limit = static_cast<int>(dimensions_[i]);
      const int value = static_cast<int>(coordinates[i]) + offsets[i];
      const int wrapped_value = ((value % limit) + limit) % limit;
      wrapped[i] = static_cast<size_type>(wrapped_value);
    }
    return wrapped;
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

  [[nodiscard]] container_type operator()(size_type orbital) const { return from_orbital(orbital); }

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
    for (size_type dim : dimensions_) {
      if (dim == 0) {
        throw std::out_of_range("Dimension with size zero is invalid.");
      }
      size *= dim;
    }
    return size;
  }
};
