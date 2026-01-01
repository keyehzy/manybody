#pragma once

#include <array>
#include <cassert>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <iterator>

#include "operator.h"

constexpr size_t term_size = 16;

struct Term {
  using complex_type = std::complex<float>;

  static constexpr size_t static_vector_size =
      (term_size - sizeof(complex_type)) / sizeof(Operator) - 1;

  template <size_t N>
  struct static_operator_vector {
    std::array<Operator, N> data{};
    uint8_t size_ = 0;

    constexpr static_operator_vector() noexcept = default;
    constexpr static_operator_vector(
        std::initializer_list<Operator> init) noexcept {
      append_range(init.begin(), init.end());
    }

    constexpr size_t size() const noexcept { return size_; }

    constexpr Operator* begin() noexcept { return data.data(); }
    constexpr const Operator* begin() const noexcept { return data.data(); }
    constexpr Operator* end() noexcept { return data.data() + size_; }
    constexpr const Operator* end() const noexcept {
      return data.data() + size_;
    }

    constexpr auto rbegin() noexcept {
      return std::reverse_iterator<Operator*>(end());
    }
    constexpr auto rbegin() const noexcept {
      return std::reverse_iterator<const Operator*>(end());
    }
    constexpr auto rend() noexcept {
      return std::reverse_iterator<Operator*>(begin());
    }
    constexpr auto rend() const noexcept {
      return std::reverse_iterator<const Operator*>(begin());
    }

    constexpr void push_back(Operator value) noexcept {
      assert(size_ < N);
      data[size_++] = value;
    }

    template <typename It>
    constexpr void append_range(It first, It last) noexcept {
      for (; first != last; ++first) {
        push_back(*first);
      }
    }

    constexpr bool operator==(
        const static_operator_vector& other) const noexcept {
      if (size_ != other.size_) {
        return false;
      }
      for (size_t i = 0; i < size_; ++i) {
        if (!(data[i] == other.data[i])) {
          return false;
        }
      }
      return true;
    }
  };

  using container_type = static_operator_vector<static_vector_size>;

  complex_type c{1.0f, 0.0f};
  container_type operators{};

  constexpr Term() noexcept = default;
  constexpr ~Term() noexcept = default;

  constexpr Term(const Term& other) noexcept = default;
  constexpr Term& operator=(const Term& other) noexcept = default;
  constexpr Term(Term&& other) noexcept = default;
  constexpr Term& operator=(Term&& other) noexcept = default;

  explicit constexpr Term(Operator x) noexcept : operators({x}) {}
  explicit constexpr Term(complex_type x) noexcept : c(x) {}
  explicit constexpr Term(const container_type& ops) noexcept
      : operators(ops) {}
  explicit constexpr Term(container_type&& ops) noexcept
      : operators(std::move(ops)) {}
  explicit constexpr Term(complex_type x, const container_type& ops) noexcept
      : c(x), operators(ops) {}
  explicit constexpr Term(complex_type x, container_type&& ops) noexcept
      : c(x), operators(std::move(ops)) {}
  explicit constexpr Term(std::initializer_list<Operator> init) noexcept
      : operators(init) {}
  explicit constexpr Term(complex_type x,
                          std::initializer_list<Operator> init) noexcept
      : c(x), operators(init) {}

  constexpr size_t size() const noexcept { return operators.size(); }

  std::string to_string() const;

  constexpr bool operator==(const Term& other) const noexcept {
    return c == other.c && operators == other.operators;
  }

  constexpr Term adjoint() const noexcept {
    Term result(std::conj(c));
    for (auto it = operators.rbegin(); it != operators.rend(); ++it) {
      result.operators.push_back(it->adjoint());
    }
    return result;
  }

  constexpr Term& operator*=(const Term& value) noexcept {
    c *= value.c;
    operators.append_range(value.operators.begin(), value.operators.end());
    return *this;
  }

  constexpr Term& operator*=(Operator value) noexcept {
    operators.push_back(value);
    return *this;
  }

  constexpr Term& operator*=(complex_type value) noexcept {
    c *= value;
    return *this;
  }

  constexpr Term& operator/=(complex_type value) noexcept {
    c /= value;
    return *this;
  }
};
static_assert(sizeof(Term) == term_size);
