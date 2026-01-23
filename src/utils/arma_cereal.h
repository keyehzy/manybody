#pragma once

#include <armadillo>
#include <cereal/cereal.hpp>
#include <cereal/types/complex.hpp>

namespace cereal {

template <class Archive, typename T>
void save(Archive& ar, const arma::SpMat<T>& t, unsigned int /* version */) {
  ar(t.n_rows, t.n_cols, t.n_nonzero);

  for (auto it = t.begin(); it != t.end(); ++it) {
    ar(it.row(), it.col(), *it);
  }
}

template <class Archive, typename T>
void load(Archive& ar, arma::SpMat<T>& t, unsigned int /* version */) {
  arma::uword r, c, nz;
  ar(r, c, nz);

  t.zeros(r, c);
  for (arma::uword i = 0; i < nz; ++i) {
    arma::uword row, col;
    T v;
    ar(row, col, v);
    t(row, col) = v;
  }
}

}  // namespace cereal
