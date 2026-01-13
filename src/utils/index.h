#pragma once

#include <cstddef>
#include <initializer_list>
#include <stdexcept>
#include <utility>
#include <vector>

class DynamicIndex {
 public:
  using size_type = std::size_t;
  using container_type = std::vector<size_type>;
  using offset_type = std::vector<int>;

  explicit DynamicIndex(container_type dimensions) { init_(std::move(dimensions)); }

  explicit DynamicIndex(std::initializer_list<size_type> dimensions) {
    init_(container_type(dimensions));
  }

  [[nodiscard]] size_type operator()(const container_type& coordinates) const {
    return to_orbital(coordinates);
  }

  [[nodiscard]] size_type operator()(std::initializer_list<size_type> coordinates) const {
    return to_orbital(container_type(coordinates));
  }

  [[nodiscard]] size_type operator()(const container_type& coordinates,
                                     const offset_type& offsets) const {
    return to_orbital_wrapped_(coordinates, offsets);
  }

  [[nodiscard]] size_type operator()(std::initializer_list<size_type> coordinates,
                                     std::initializer_list<int> offsets) const {
    return to_orbital_wrapped_(container_type(coordinates), offset_type(offsets));
  }

  [[nodiscard]] container_type operator()(size_type orbital) const { return from_orbital(orbital); }

  [[nodiscard]] size_type value_at(size_type orbital, size_type dimension) const {
    if (dimension >= dimensions_.size()) {
      throw std::out_of_range("Dimension is out of bounds.");
    }
    if (orbital >= total_size_) {
      throw std::out_of_range("Orbital index out of bounds.");
    }
    return (orbital / strides_[dimension]) % dimensions_[dimension];
  }

  const container_type& dimensions() const noexcept { return dimensions_; }

  size_type dimension(size_type index) const {
    if (index >= dimensions_.size()) {
      throw std::out_of_range("Dimension is out of bounds.");
    }
    return dimensions_[index];
  }

  [[nodiscard]] size_type size() const { return total_size_; }

 private:
  container_type dimensions_;
  container_type strides_;
  size_type total_size_{0};

  void init_(container_type dims) {
    if (dims.empty()) {
      throw std::out_of_range("Empty dimensions are invalid.");
    }
    for (size_type d : dims) {
      if (d == 0) {
        throw std::out_of_range("Dimension with size zero is invalid.");
      }
    }

    dimensions_ = std::move(dims);
    strides_.resize(dimensions_.size());

    size_type stride = 1;
    for (size_type i = 0; i < dimensions_.size(); ++i) {
      strides_[i] = stride;
      stride *= dimensions_[i];
    }
    total_size_ = stride;
  }

  [[nodiscard]] size_type to_orbital(const container_type& coordinates) const {
    if (coordinates.size() != dimensions_.size()) {
      throw std::out_of_range("Invalid number of coordinates.");
    }

    size_type orbital = 0;
    for (size_type i = 0; i < dimensions_.size(); ++i) {
      if (coordinates[i] >= dimensions_[i]) {
        throw std::out_of_range("Coordinates out of bounds.");
      }
      orbital += coordinates[i] * strides_[i];
    }
    return orbital;
  }

  [[nodiscard]] container_type from_orbital(size_type orbital) const {
    if (orbital >= total_size_) {
      throw std::out_of_range("Orbital index out of bounds.");
    }

    container_type coordinates(dimensions_.size());
    for (size_type i = 0; i < dimensions_.size(); ++i) {
      coordinates[i] = (orbital / strides_[i]) % dimensions_[i];
    }
    return coordinates;
  }

  [[nodiscard]] size_type to_orbital_wrapped_(const container_type& coordinates,
                                              const offset_type& offsets) const {
    if (coordinates.size() != dimensions_.size()) {
      throw std::out_of_range("Invalid number of coordinates.");
    }
    if (offsets.size() != dimensions_.size()) {
      throw std::out_of_range("Invalid number of offsets.");
    }

    size_type orbital = 0;
    for (size_type i = 0; i < dimensions_.size(); ++i) {
      const auto dim_sz = dimensions_[i];
      if (coordinates[i] >= dim_sz) {
        throw std::out_of_range("Coordinates out of bounds.");
      }

      const int dim = static_cast<int>(dim_sz);
      int v = static_cast<int>(coordinates[i]) + offsets[i];
      v %= dim;
      if (v < 0) {
        v += dim;
      }

      orbital += static_cast<size_type>(v) * strides_[i];
    }
    return orbital;
  }
};
