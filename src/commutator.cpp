#include "commutator.h"

#include "normal_order.h"

Expression commutator(const Term& A, const Term& B) {
  NormalOrderer orderer;
  Expression result = orderer.normal_order(A * B);
  result -= orderer.normal_order(B * A);
  return result;
}

Expression commutator(const Expression& A, const Expression& B) {
  NormalOrderer orderer;
  Expression result = orderer.normal_order(A * B);
  result -= orderer.normal_order(B * A);
  return result;
}

Expression anticommutator(const Term& A, const Term& B) {
  NormalOrderer orderer;
  Expression result = orderer.normal_order(A * B);
  result += orderer.normal_order(B * A);
  return result;
}

Expression anticommutator(const Expression& A, const Expression& B) {
  NormalOrderer orderer;
  Expression result = orderer.normal_order(A * B);
  result += orderer.normal_order(B * A);
  return result;
}
