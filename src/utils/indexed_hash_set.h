#pragma once

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <functional>
#include <initializer_list>
#include <iterator>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include "robin_hood.h"

#ifndef NDEBUG
#include <unordered_set>
#endif

template <typename Key, typename Hash = std::hash<Key>, typename KeyEqual = std::equal_to<Key>>
class IndexedHashSet {
 public:
  using key_type = Key;
  using value_type = Key;
  using size_type = std::size_t;
  using difference_type = std::ptrdiff_t;
  using hasher = Hash;
  using key_equal = KeyEqual;
  using reference = const value_type&;
  using const_reference = const value_type&;
  using pointer = const value_type*;
  using const_pointer = const value_type*;
  using iterator = typename std::vector<Key>::const_iterator;
  using const_iterator = typename std::vector<Key>::const_iterator;
  using reverse_iterator = std::reverse_iterator<iterator>;
  using const_reverse_iterator = std::reverse_iterator<const_iterator>;

 private:
  std::vector<Key> elements_;
  robin_hood::unordered_map<Key, size_type, Hash, KeyEqual> indices_;

  void build_index_map() {
    indices_.reserve(elements_.size());
    for (size_type i = 0; i < elements_.size(); ++i) {
      indices_.emplace(elements_[i], i);
    }
  }

  template <typename InputIt>
  void initialize_members_from_range(InputIt first, InputIt last) {
#ifndef NDEBUG
    std::unordered_set<Key, Hash, KeyEqual> seen_keys;
#endif
    std::vector<Key> unique_elements;

    size_type initial_capacity = 0;
    if constexpr (std::is_base_of_v<std::forward_iterator_tag,
                                    typename std::iterator_traits<InputIt>::iterator_category>) {
      initial_capacity = static_cast<size_type>(std::distance(first, last));
#ifndef NDEBUG
      seen_keys.reserve(initial_capacity);
#endif
      unique_elements.reserve(initial_capacity);
    }

    for (; first != last; ++first) {
      const Key& current_key = *first;
#ifndef NDEBUG
      auto [_, inserted_successfully] = seen_keys.insert(current_key);
      if (!inserted_successfully) {
        throw std::invalid_argument("Duplicate key found in input.");
      }
#endif
      unique_elements.push_back(current_key);
    }

    elements_ = std::move(unique_elements);
    elements_.shrink_to_fit();
    build_index_map();
  }

  void initialize_members_from_moved_vector(std::vector<Key>&& source_vector) {
#ifndef NDEBUG
    std::unordered_set<Key, Hash, KeyEqual> seen_keys;
#endif
    std::vector<Key> unique_elements;

#ifndef NDEBUG
    seen_keys.reserve(source_vector.size());
#endif
    unique_elements.reserve(source_vector.size());

    for (Key& element : source_vector) {
#ifndef NDEBUG
      auto [_, inserted_successfully] = seen_keys.insert(element);
      if (!inserted_successfully) {
        throw std::invalid_argument("Duplicate key found in input.");
      }
#endif
      unique_elements.push_back(std::move(element));
    }

    elements_ = std::move(unique_elements);
    elements_.shrink_to_fit();
    build_index_map();
  }

 public:
  IndexedHashSet() noexcept(std::is_nothrow_default_constructible_v<std::vector<Key>> &&
                            std::is_nothrow_default_constructible_v<
                                robin_hood::unordered_map<Key, size_type, Hash, KeyEqual>>)
      : elements_{}, indices_{} {}

  explicit IndexedHashSet(std::initializer_list<Key> init_list) : elements_{}, indices_{} {
    initialize_members_from_range(init_list.begin(), init_list.end());
  }

  template <typename InputIt>
  IndexedHashSet(InputIt first, InputIt last) : elements_{}, indices_{} {
    initialize_members_from_range(first, last);
  }

  explicit IndexedHashSet(const std::vector<Key>& source_vector) : elements_{}, indices_{} {
    initialize_members_from_range(source_vector.begin(), source_vector.end());
  }

  explicit IndexedHashSet(std::vector<Key>&& source_vector) : elements_{}, indices_{} {
    initialize_members_from_moved_vector(std::move(source_vector));
  }

  IndexedHashSet(const IndexedHashSet&) = default;
  IndexedHashSet(IndexedHashSet&&) noexcept = default;
  IndexedHashSet& operator=(const IndexedHashSet&) = default;
  IndexedHashSet& operator=(IndexedHashSet&&) noexcept = default;

  bool empty() const noexcept { return elements_.empty(); }

  size_type size() const noexcept { return elements_.size(); }

  size_type max_size() const noexcept {
    return std::min(elements_.max_size(), indices_.max_size());
  }

  bool contains(const Key& key) const { return indices_.count(key) > 0; }

  const_reference at(size_type index) const {
    if (index >= elements_.size()) {
      throw std::out_of_range("Index out of range.");
    }
    return elements_[index];
  }

  const_reference operator[](size_type index) const {
    assert(index < elements_.size());
    return elements_[index];
  }

  size_type index_of(const Key& key) const {
    auto map_iterator = indices_.find(key);
    if (map_iterator == indices_.end()) {
      throw std::out_of_range("Index out of range.");
    }
    return map_iterator->second;
  }

  const std::vector<Key>& elements() const { return elements_; }

  const_iterator begin() const noexcept { return elements_.cbegin(); }
  const_iterator end() const noexcept { return elements_.cend(); }
  const_iterator cbegin() const noexcept { return elements_.cbegin(); }
  const_iterator cend() const noexcept { return elements_.cend(); }

  const_reverse_iterator rbegin() const noexcept { return elements_.crbegin(); }
  const_reverse_iterator rend() const noexcept { return elements_.crend(); }
  const_reverse_iterator crbegin() const noexcept { return elements_.crbegin(); }
  const_reverse_iterator crend() const noexcept { return elements_.crend(); }

  friend bool operator==(const IndexedHashSet& lhs, const IndexedHashSet& rhs) {
    if (lhs.size() != rhs.size()) {
      return false;
    }
    return std::equal(lhs.elements_.begin(), lhs.elements_.end(), rhs.elements_.begin());
  }

  friend bool operator!=(const IndexedHashSet& lhs, const IndexedHashSet& rhs) {
    return !(lhs == rhs);
  }
};
