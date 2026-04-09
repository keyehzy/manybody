#pragma once

#include <complex>
#include <cstddef>

#include "algebra/monomial.h"
#include "algebra/operator.h"
#include "algebra/term_utils.h"

using BosonScalar = std::complex<double>;

constexpr size_t boson_term_size = 32;

constexpr size_t boson_term_static_vector_size =
    (boson_term_size - sizeof(BosonScalar)) / sizeof(BosonOperator) - 1;

using BosonMonomial =
    MonomialImpl<BosonOperator, boson_term_static_vector_size, BosonOperator::ubyte, BosonScalar>;

namespace boson {

inline constexpr BosonMonomial creation(BosonOperator::Spin spin, size_t orbital) noexcept {
  return detail::make_creation_monomial<BosonMonomial>(spin, orbital);
}

inline constexpr BosonMonomial annihilation(BosonOperator::Spin spin, size_t orbital) noexcept {
  return detail::make_annihilation_monomial<BosonMonomial>(spin, orbital);
}

inline constexpr BosonMonomial one_body(BosonOperator::Spin s1, size_t o1, BosonOperator::Spin s2,
                                        size_t o2) noexcept {
  return detail::make_one_body_monomial<BosonMonomial>(s1, o1, s2, o2);
}

inline constexpr BosonMonomial two_body(BosonOperator::Spin s1, size_t o1, BosonOperator::Spin s2,
                                        size_t o2, BosonOperator::Spin s3, size_t o3,
                                        BosonOperator::Spin s4, size_t o4) noexcept {
  return detail::make_two_body_monomial<BosonMonomial>(s1, o1, s2, o2, s3, o3, s4, o4);
}

inline constexpr BosonMonomial number_op(BosonOperator::Spin s, size_t o) noexcept {
  return detail::make_number_monomial<BosonMonomial>(s, o);
}

}  // namespace boson

static_assert(sizeof(BosonMonomial) == boson_term_size);
