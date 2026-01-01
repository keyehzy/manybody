#pragma once

#include <array>
#include <cassert>
#include <cstddef>
#include <initializer_list>
#include <iterator>

template <typename T, size_t N, typename SizeType = std::size_t>
struct static_vector {
  std::array<T, N> data{};
  SizeType size_ = 0;

  constexpr static_vector() noexcept = default;
  constexpr static_vector(std::initializer_list<T> init) noexcept {
    append_range(init.begin(), init.end());
  }

  constexpr size_t size() const noexcept { return static_cast<size_t>(size_); }

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

  constexpr auto rbegin() noexcept { return std::reverse_iterator<T*>(end()); }
  constexpr auto rbegin() const noexcept {
    return std::reverse_iterator<const T*>(end());
  }
  constexpr auto rend() noexcept { return std::reverse_iterator<T*>(begin()); }
  constexpr auto rend() const noexcept {
    return std::reverse_iterator<const T*>(begin());
  }

  constexpr void push_back(const T& value) noexcept {
    assert(size_ < N);
    data[size_++] = value;
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
