#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <functional>
#include <initializer_list>
#include <iterator>
#include <type_traits>

#include "xxhash.h"

template <typename T, size_t N, typename SizeType = std::size_t>
struct static_vector {
  std::array<T, N> data{};
  SizeType size_ = 0;

  constexpr static_vector() noexcept = default;
  constexpr static_vector(std::initializer_list<T> init) noexcept {
    append_range(init.begin(), init.end());
  }

  static constexpr size_t max_size() noexcept { return N; }
  constexpr size_t size() const noexcept { return static_cast<size_t>(size_); }
  constexpr bool empty() const noexcept { return size_ == 0; }

  constexpr void clear() noexcept { size_ = 0; }

  constexpr T* begin() noexcept { return data.data(); }
  constexpr const T* begin() const noexcept { return data.data(); }
  constexpr T* end() noexcept { return data.data() + size(); }
  constexpr const T* end() const noexcept { return data.data() + size(); }

  constexpr T& operator[](size_t index) noexcept {
    assert(index < size());
    return data[index];
  }
  constexpr const T& operator[](size_t index) const noexcept {
    assert(index < size());
    return data[index];
  }

  constexpr T& at(size_t index) noexcept {
    assert(index < size());
    return data[index];
  }
  constexpr const T& at(size_t index) const noexcept {
    assert(index < size());
    return data[index];
  }

  constexpr T& back() noexcept {
    assert(size_ > 0);
    return data[size_ - 1];
  }
  constexpr const T& back() const noexcept {
    assert(size_ > 0);
    return data[size_ - 1];
  }

  size_t hash() const noexcept {
    static_assert(std::is_trivially_copyable_v<T>,
                  "static_vector::hash requires trivially copyable elements");
    const XXH64_hash_t seed = XXH64(&size_, sizeof(size_), 0);
    return static_cast<size_t>(XXH64(data.data(), size() * sizeof(T), seed));
  }

  constexpr auto rbegin() noexcept { return std::reverse_iterator<T*>(end()); }
  constexpr auto rbegin() const noexcept { return std::reverse_iterator<const T*>(end()); }
  constexpr auto rend() noexcept { return std::reverse_iterator<T*>(begin()); }
  constexpr auto rend() const noexcept { return std::reverse_iterator<const T*>(begin()); }

  constexpr void push_back(const T& value) noexcept {
    assert(size_ < N);
    data[size_++] = value;
  }

  constexpr void resize(size_t count) noexcept {
    assert(count <= N);
    if (count > size()) {
      for (size_t i = size(); i < count; ++i) {
        data[i] = T{};
      }
    }
    size_ = static_cast<SizeType>(count);
  }

  constexpr void resize(size_t count, const T& value) noexcept {
    assert(count <= N);
    if (count > size()) {
      for (size_t i = size(); i < count; ++i) {
        data[i] = value;
      }
    }
    size_ = static_cast<SizeType>(count);
  }

  constexpr void pop_back() noexcept {
    assert(size_ > 0);
    --size_;
  }

  constexpr void insert(size_t pos, const T& value) noexcept {
    assert(pos <= size());
    assert(size_ < N);
    for (size_t i = size(); i > pos; --i) {
      data[i] = data[i - 1];
    }
    data[pos] = value;
    ++size_;
  }

  constexpr void erase(size_t pos) noexcept {
    assert(pos < size());
    for (size_t i = pos; i + 1 < size(); ++i) {
      data[i] = data[i + 1];
    }
    --size_;
  }

  template <typename It>
  constexpr void append_range(It first, It last) noexcept {
    for (; first != last; ++first) {
      push_back(*first);
    }
  }

  constexpr bool operator==(const static_vector& other) const noexcept {
    if (size_ != other.size_) {
      return false;
    }
    for (size_t i = 0; i < size(); ++i) {
      if (!(data[i] == other.data[i])) {
        return false;
      }
    }
    return true;
  }
};

template <typename T, size_t N, typename SizeType>
struct std::hash<static_vector<T, N, SizeType>> {
  size_t operator()(const static_vector<T, N, SizeType>& value) const noexcept {
    return value.hash();
  }
};

template <typename T, size_t N, typename SizeType>
constexpr bool has_consecutive_elements(const static_vector<T, N, SizeType>& container) noexcept {
  return std::adjacent_find(container.begin(), container.end()) != container.end();
}
