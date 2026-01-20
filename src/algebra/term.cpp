#include "algebra/term.h"

void Term::to_string(std::ostringstream& oss) const {
  if (c == complex_type{}) {
    oss << "0";
    return;
  }
  oss << c;
  if (operators.size() == 0) {
    return;
  }
  oss << " ";
  for (const auto& op : operators) {
    op.to_string(oss);
  }
  if (is_diagonal()) {
    oss << "*";
  }
}

std::string Term::to_string() const {
  std::ostringstream oss;
  to_string(oss);
  return oss.str();
}
