#pragma once

#include <algorithm>
#include <complex>
#include <sstream>
#include <utility>
#include <vector>

#include "robin_hood.h"
#include "utils/tolerances.h"

template <typename Derived, typename MonomialType>
struct ExpressionBase {
  using complex_type = typename MonomialType::complex_type;
  using container_type = typename MonomialType::container_type;
  using map_type = robin_hood::unordered_map<container_type, complex_type>;

  map_type data{};

  const map_type& terms() const noexcept { return data; }
  map_type& terms() noexcept { return data; }

  ExpressionBase() = default;
  ~ExpressionBase() = default;

  ExpressionBase(const ExpressionBase&) = default;
  ExpressionBase& operator=(const ExpressionBase&) = default;
  ExpressionBase(ExpressionBase&&) noexcept = default;
  ExpressionBase& operator=(ExpressionBase&&) noexcept = default;

  explicit ExpressionBase(complex_type c) {
    if (!is_zero(c)) {
      data.emplace(container_type{}, c);
    }
  }

  explicit ExpressionBase(const container_type& container) {
    data.emplace(container, complex_type{1.0, 0.0});
  }

  explicit ExpressionBase(container_type&& container) {
    data.emplace(std::move(container), complex_type{1.0, 0.0});
  }

  template <typename Container>
  ExpressionBase(complex_type c, Container&& ops) {
    data.emplace(std::forward<Container>(ops), c);
  }

  size_t size() const { return data.size(); }
  bool empty() const { return data.empty(); }
  void clear() { data.clear(); }
  void reserve(size_t n) { data.reserve(n); }

  static bool is_zero(const complex_type& value) {
    using value_type = typename complex_type::value_type;
    constexpr auto tolerance = tolerances::tolerance<value_type>();
    return std::norm(value) < tolerance * tolerance;
  }

  static void add_to(map_type& target, const container_type& key, const complex_type& coeff) {
    if (is_zero(coeff)) {
      return;
    }
    auto [it, inserted] = target.try_emplace(key, coeff);
    if (!inserted) {
      it->second += coeff;
      if (is_zero(it->second)) {
        target.erase(it);
      }
    }
  }

  static void add_to(map_type& target, container_type&& key, const complex_type& coeff) {
    if (is_zero(coeff)) {
      return;
    }
    auto [it, inserted] = target.try_emplace(std::move(key), coeff);
    if (!inserted) {
      it->second += coeff;
      if (is_zero(it->second)) {
        target.erase(it);
      }
    }
  }

  void add(const container_type& key, const complex_type& coeff) { add_to(data, key, coeff); }

  void add(container_type&& key, const complex_type& coeff) {
    add_to(data, std::move(key), coeff);
  }

  void add_scalar(const complex_type& value) { add(container_type{}, value); }

  void subtract_scalar(const complex_type& value) { add(container_type{}, -value); }

  void scale(const complex_type& value) {
    if (is_zero(value)) {
      data.clear();
      return;
    }
    for (auto& [key, coeff] : data) {
      coeff *= value;
    }
  }

  void divide(const complex_type& value) {
    for (auto& [key, coeff] : data) {
      coeff /= value;
    }
  }

  void add_all(const ExpressionBase& other) {
    for (const auto& [key, coeff] : other.data) {
      add(key, coeff);
    }
  }

  void subtract_all(const ExpressionBase& other) {
    for (const auto& [key, coeff] : other.data) {
      add(key, -coeff);
    }
  }

  Derived& operator+=(const complex_type& v) {
    add_scalar(v);
    return static_cast<Derived&>(*this);
  }
  Derived& operator-=(const complex_type& v) {
    subtract_scalar(v);
    return static_cast<Derived&>(*this);
  }
  Derived& operator*=(const complex_type& v) {
    scale(v);
    return static_cast<Derived&>(*this);
  }
  Derived& operator/=(const complex_type& v) {
    divide(v);
    return static_cast<Derived&>(*this);
  }
  Derived& operator+=(const ExpressionBase& o) {
    add_all(o);
    return static_cast<Derived&>(*this);
  }
  Derived& operator-=(const ExpressionBase& o) {
    subtract_all(o);
    return static_cast<Derived&>(*this);
  }

  void truncate_by_norm(double min_norm) {
    if (min_norm <= 0.0) {
      return;
    }
    using value_type = typename complex_type::value_type;
    const auto cutoff = static_cast<value_type>(min_norm);
    const auto cutoff_norm = cutoff * cutoff;
    for (auto it = data.begin(); it != data.end();) {
      if (std::norm(it->second) < cutoff_norm) {
        it = data.erase(it);
      } else {
        ++it;
      }
    }
  }

  template <class FormatEntry>
  std::string to_string(FormatEntry fmt) const {
    std::ostringstream oss;
    format_sorted(oss, fmt);
    return oss.str();
  }

  template <class FormatEntry>
  void format_sorted(std::ostringstream& oss, FormatEntry fmt) const {
    if (data.empty()) {
      oss << "0";
      return;
    }
    std::vector<const typename map_type::value_type*> ordered;
    ordered.reserve(data.size());
    for (const auto& entry : data) {
      ordered.push_back(&entry);
    }
    std::sort(
        ordered.begin(), ordered.end(),
        [](const typename map_type::value_type* left, const typename map_type::value_type* right) {
          const auto left_size = left->first.size();
          const auto right_size = right->first.size();
          if (left_size != right_size) {
            return left_size < right_size;
          }
          return std::norm(left->second) > std::norm(right->second);
        });

    bool first = true;
    for (const auto* entry : ordered) {
      if (!first) {
        oss << "\n";
      }
      fmt(oss, entry->first, entry->second);
      first = false;
    }
  }

  Derived& operator+=(const MonomialType& value) {
    static_cast<Derived*>(this)->add_to_map(value.operators, value.c);
    return static_cast<Derived&>(*this);
  }

  Derived& operator-=(const MonomialType& value) {
    static_cast<Derived*>(this)->add_to_map(value.operators, -value.c);
    return static_cast<Derived&>(*this);
  }
};
