#include "algebra/term.h"

void to_string(std::ostringstream& oss, const Term& term) {
  if (term.c == Term::complex_type{}) {
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

std::string to_string(const Term& term) {
  std::ostringstream oss;
  to_string(oss, term);
  return oss.str();
}
