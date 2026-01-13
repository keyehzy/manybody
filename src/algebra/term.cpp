#include "algebra/term.h"

#include <sstream>

std::string Term::to_string() const {
  if (c == complex_type{}) {
    return "0";
  }
  std::ostringstream oss;
  oss << c;
  if (operators.size() == 0) {
    return oss.str();
  }
  oss << " ";
  for (const auto& op : operators) {
    oss << op.to_string();
  }
  return oss.str();
}
