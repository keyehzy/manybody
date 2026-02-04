#pragma once

#include <algorithm>
#include <complex>
#include <sstream>
#include <utility>
#include <vector>

#include "robin_hood.h"
#include "utils/tolerances.h"

template <class Key>
struct ExpressionMap {
  using complex_type = std::complex<double>;
  using key_type = Key;
  using map_type = robin_hood::unordered_map<Key, complex_type>;

  map_type data{};

  size_t size() const { return data.size(); }
  bool empty() const { return data.empty(); }
  void clear() { data.clear(); }
  void reserve(size_t n) { data.reserve(n); }

  static bool is_zero(const complex_type& value) {
    constexpr auto tolerance = tolerances::tolerance<complex_type::value_type>();
    return std::norm(value) < tolerance * tolerance;
  }

  void add(const key_type& key, const complex_type& coeff) {
    if (is_zero(coeff)) {
      return;
    }
    auto [it, inserted] = data.try_emplace(key, coeff);
    if (!inserted) {
      it->second += coeff;
      if (is_zero(it->second)) {
        data.erase(it);
      }
    }
  }

  void add(key_type&& key, const complex_type& coeff) {
    if (is_zero(coeff)) {
      return;
    }
    auto [it, inserted] = data.try_emplace(std::move(key), coeff);
    if (!inserted) {
      it->second += coeff;
      if (is_zero(it->second)) {
        data.erase(it);
      }
    }
  }

  void add_scalar(const complex_type& value) { add(key_type{}, value); }

  void subtract_scalar(const complex_type& value) { add(key_type{}, -value); }

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

  void add_all(const ExpressionMap& other) {
    for (const auto& [key, coeff] : other.data) {
      add(key, coeff);
    }
  }

  void subtract_all(const ExpressionMap& other) {
    for (const auto& [key, coeff] : other.data) {
      add(key, -coeff);
    }
  }

  ExpressionMap& operator+=(const complex_type& v) { add_scalar(v); return *this; }
  ExpressionMap& operator-=(const complex_type& v) { subtract_scalar(v); return *this; }
  ExpressionMap& operator*=(const complex_type& v) { scale(v); return *this; }
  ExpressionMap& operator/=(const complex_type& v) { divide(v); return *this; }
  ExpressionMap& operator+=(const ExpressionMap& o) { add_all(o); return *this; }
  ExpressionMap& operator-=(const ExpressionMap& o) { subtract_all(o); return *this; }

  void truncate_by_norm(double min_norm) {
    if (min_norm <= 0.0) {
      return;
    }
    const auto cutoff = static_cast<complex_type::value_type>(min_norm);
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
};
