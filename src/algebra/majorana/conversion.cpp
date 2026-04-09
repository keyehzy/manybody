#include "algebra/majorana/conversion.h"

namespace majorana {

/// Convert a fermionic FermionExpression (in terms of c, c+) to a MajoranaExpression.
///
/// For each fermionic mode (orbital, spin) with Majorana indices (e, o):
///   c+  = (gamma_e + i * gamma_o) / 2
///   c   = (gamma_e - i * gamma_o) / 2
///
/// Each operator in a term is substituted with the corresponding sum of two
/// Majorana strings, and the resulting products are expanded.
MajoranaExpression to_majorana(const FermionExpression& expr) {
  using complex_type = MajoranaExpression::complex_type;

  MajoranaExpression result;

  for (const auto& [ops, coeff] : expr.terms()) {
    // Start with the scalar coefficient as a MajoranaExpression
    MajoranaExpression term_expr(coeff);

    for (size_t k = 0; k < ops.size(); ++k) {
      const FermionOperator op = ops[k];
      MajoranaMonomial::container_type even_str;
      MajoranaMonomial::container_type odd_str;
      even_str.push_back(MajoranaOperator::even(op.value(), op.spin()));
      odd_str.push_back(MajoranaOperator::odd(op.value(), op.spin()));

      // c+  = (gamma_e + i * gamma_o) / 2
      // c   = (gamma_e - i * gamma_o) / 2
      MajoranaExpression op_expr;
      if (op.type() == FermionOperator::Type::Creation) {
        op_expr += MajoranaExpression(complex_type(0.5, 0.0), even_str);
        op_expr += MajoranaExpression(complex_type(0.0, 0.5), odd_str);
      } else {
        op_expr += MajoranaExpression(complex_type(0.5, 0.0), even_str);
        op_expr += MajoranaExpression(complex_type(0.0, -0.5), odd_str);
      }

      term_expr *= op_expr;
    }

    result += term_expr;
  }

  return result;
}

/// Convert a MajoranaExpression back to a fermionic FermionExpression.
///
/// For each Majorana index, reconstruct the fermionic operator:
///   gamma_e = c + c+
///   gamma_o = -i(c - c+) = i(c+ - c)
///
/// The result is normal-ordered.
FermionExpression from_majorana(const MajoranaExpression& expr) {
  using complex_type = FermionExpression::complex_type;
  FermionExpression result;

  for (const auto& [str, coeff] : expr.terms()) {
    FermionExpression term_expr(coeff);

    for (const auto& element : str) {
      const size_t orbital = element.orbital();
      const auto spin = element.spin();

      FermionOperator create = FermionOperator::creation(spin, orbital);
      FermionOperator annihilate = FermionOperator::annihilation(spin, orbital);

      FermionExpression op_expr;
      if (element.is_even()) {
        // gamma_e = c + c+
        op_expr += FermionExpression(create);
        op_expr += FermionExpression(annihilate);
      } else {
        // gamma_o = -i(c+ - c) = -i*c+ + i*c
        op_expr += FermionExpression(FermionMonomial(complex_type(0.0, -1.0), {create}));
        op_expr += FermionExpression(FermionMonomial(complex_type(0.0, 1.0), {annihilate}));
      }

      term_expr *= op_expr;
    }

    result += term_expr;
  }

  return canonicalize(result);
}

}  // namespace majorana
