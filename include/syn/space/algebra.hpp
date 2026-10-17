#pragma once

// The whole algebra of the space: multivectors of Cl(4, 1), for the models of functional
// geometric algebra (Pustejovsky, "Toward a Functional Geometric Algebra for Natural Language
// Semantics", 2026). The space's points are its null vectors (syn/space/space.hpp); here every
// grade is available — scalars (truth values), vectors (entities, predicates, roles), bivectors
// (relations, role–filler bindings), trivectors and up — with the geometric product, the left and
// right contractions, the wedge, the reverse and grade projection, written once in WGSL.
//
// A multivector has 32 coefficients, one per basis blade; a blade is a bit mask over the basis
// vectors (bit 0 N, 1 A, 2 B, 3 C, 4 e; e² = −1, the others square to +1). Multivectors live in
// the slots of a private bank, and the products read their signs from the table e_a e_b =
// sign(a, b)·e_{a ⊕ b}, which a model puts at the start of its constants: P(0..1023) is the
// table and Q(i) = P(1024 + i) the model's own constants.

#include <string>
#include <vector>

namespace syn::space {

/** Blades of Cl(4, 1) and the size of the sign table at the start of a model's constants. */
constexpr unsigned kBlades = 32;
constexpr unsigned kSignTable = kBlades * kBlades;

/** The signs of the geometric product of basis blades, row-major (a · 32 + b). */
std::vector<double> signTable();

/** The table followed by a model's own constants (read in WGSL as Q(i)). */
std::vector<double> withSignTable(const std::vector<double>& constants);

/** WGSL: Q(i), the bank MV of `slots` multivectors and the operations of the algebra on it. */
std::string algebraWgsl(unsigned slots);

}  // namespace syn::space
