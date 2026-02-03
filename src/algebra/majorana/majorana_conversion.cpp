#include "algebra/majorana/majorana_conversion.h"

#include "algebra/normal_order.h"

/// Convert a fermionic Expression (in terms of c, c+) to a MajoranaExpression.
///
/// For each fermionic mode (orbital, spin) with Majorana indices (e, o):
///   c+  = (gamma_e + i * gamma_o) / 2
///   c   = (gamma_e - i * gamma_o) / 2
///
/// Each operator in a term is substituted with the corresponding sum of two
/// Majorana strings, and the resulting products are expanded.
MajoranaExpression to_majorana(const Expression& expr) {
  using complex_type = MajoranaExpression::complex_type;

  MajoranaExpression result;

  for (const auto& [ops, coeff] : expr.hashmap) {
    // Start with the scalar coefficient as a MajoranaExpression
    MajoranaExpression term_expr(coeff);

    for (size_t k = 0; k < ops.size(); ++k) {
      const Operator op = ops[k];
      auto [even_idx, odd_idx] = majorana_indices(op);

      MajoranaString even_str{even_idx};
      MajoranaString odd_str{odd_idx};

      // c+  = (gamma_e + i * gamma_o) / 2
      // c   = (gamma_e - i * gamma_o) / 2
      MajoranaExpression op_expr;
      if (op.type() == Operator::Type::Creation) {
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

/// Convert a MajoranaExpression back to a fermionic Expression.
///
/// For each Majorana index, reconstruct the fermionic operator:
///   gamma_e = c + c+
///   gamma_o = -i(c - c+) = i(c+ - c)
///
/// The result is normal-ordered.
Expression from_majorana(const MajoranaExpression& expr) {
  using complex_type = Expression::complex_type;
  NormalOrderer orderer;
  Expression result;

  for (const auto& [str, coeff] : expr.hashmap) {
    Expression term_expr(coeff);

    for (size_t k = 0; k < str.size(); ++k) {
      const MajoranaIndex idx = str[k];
      // Decode: flat = 4 * orbital + 2 * parity + spin_bit
      const size_t orbital = idx / 4;
      const auto parity = static_cast<size_t>((idx / 2) % 2);
      const auto spin_bit = static_cast<size_t>(idx % 2);
      const auto spin = (spin_bit == 0) ? Operator::Spin::Up : Operator::Spin::Down;

      Operator create = Operator::creation(spin, orbital);
      Operator annihilate = Operator::annihilation(spin, orbital);

      Expression op_expr;
      if (parity == 0) {
        // gamma_e = c + c+
        op_expr += Expression(create);
        op_expr += Expression(annihilate);
      } else {
        // gamma_o = -i(c+ - c) = -i*c+ + i*c
        op_expr += Expression(Term(complex_type(0.0, -1.0), {create}));
        op_expr += Expression(Term(complex_type(0.0, 1.0), {annihilate}));
      }

      term_expr *= op_expr;
    }

    result += term_expr;
  }

  return orderer.normal_order(result);
}
