#include "algebra/term.h"

void to_string(std::ostringstream& oss, const FermionMonomial& term) {
  if (term.c == FermionMonomial::complex_type{}) {
    oss << "0";
    return;
  }
  oss << term.c;
  if (term.operators.size() == 0) {
    return;
  }
  oss << " ";
  for (const auto& op : term.operators) {
    op.to_string(oss);
  }
  if (is_diagonal(term)) {
    oss << "*";
  }
}

std::string to_string(const FermionMonomial& term) {
  std::ostringstream oss;
  to_string(oss, term);
  return oss.str();
}
