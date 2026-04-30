#pragma once

#include <complex>
#include <cstddef>

#include "algebra/monomial.h"
#include "algebra/operator.h"
#include "algebra/term_utils.h"

using HardcoreBosonScalar = std::complex<double>;

constexpr size_t hardcore_boson_term_size = 32;

constexpr size_t hardcore_boson_term_static_vector_size =
    (hardcore_boson_term_size - sizeof(HardcoreBosonScalar)) / sizeof(HardcoreBosonOperator) - 1;

using HardcoreBosonMonomial =
    MonomialImpl<HardcoreBosonOperator, hardcore_boson_term_static_vector_size,
                 HardcoreBosonOperator::ubyte, HardcoreBosonScalar>;

namespace hardcore_boson {

inline constexpr HardcoreBosonMonomial creation(HardcoreBosonOperator::Spin spin,
                                                size_t orbital) noexcept {
  return detail::make_creation_monomial<HardcoreBosonMonomial>(spin, orbital);
}

inline constexpr HardcoreBosonMonomial annihilation(HardcoreBosonOperator::Spin spin,
                                                    size_t orbital) noexcept {
  return detail::make_annihilation_monomial<HardcoreBosonMonomial>(spin, orbital);
}

inline constexpr HardcoreBosonMonomial one_body(HardcoreBosonOperator::Spin s1, size_t o1,
                                                HardcoreBosonOperator::Spin s2,
                                                size_t o2) noexcept {
  return detail::make_one_body_monomial<HardcoreBosonMonomial>(s1, o1, s2, o2);
}

inline constexpr HardcoreBosonMonomial two_body(HardcoreBosonOperator::Spin s1, size_t o1,
                                                HardcoreBosonOperator::Spin s2, size_t o2,
                                                HardcoreBosonOperator::Spin s3, size_t o3,
                                                HardcoreBosonOperator::Spin s4,
                                                size_t o4) noexcept {
  return detail::make_two_body_monomial<HardcoreBosonMonomial>(s1, o1, s2, o2, s3, o3, s4, o4);
}

inline constexpr HardcoreBosonMonomial number_op(HardcoreBosonOperator::Spin s,
                                                 size_t o) noexcept {
  return detail::make_number_monomial<HardcoreBosonMonomial>(s, o);
}

inline constexpr HardcoreBosonMonomial density_density(HardcoreBosonOperator::Spin s1, size_t i,
                                                       HardcoreBosonOperator::Spin s2,
                                                       size_t j) noexcept {
  return number_op(s1, i) * number_op(s2, j);
}

}  // namespace hardcore_boson

static_assert(sizeof(HardcoreBosonMonomial) == hardcore_boson_term_size);
