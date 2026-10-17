// Functional geometric algebra in the space: the claims of Pustejovsky (2026), checked in Cl(4, 1).
//
// The paper proposes Clifford algebras as the substrate of compositional semantics: grades as
// types (scalars truth values, vectors entities and predicates, bivectors relations), the left and
// right contractions as function application (Layer 1), the wedge as role–filler binding with
// exact unbinding (Layer 2), rotors as coercion (Layer 3), and idempotents ½(1 ± u) as polar
// opposition. Here every one of those operations runs in the space itself. The base space
// decomposes as V_E = ⟨A, B, C⟩ (entities: points of a 2-sphere), V_P = ⟨e⟩ (predicate content,
// purpose: the paper's negative-squaring direction, e² = −1) and the roles r_AG = N, r_TH = e.
// Agents are random lexicons: each draws its entities, relation, predicate and test vectors, runs
// the derivations of the paper and adds the error of each claim. Most claims hold exactly (to
// 32-bit rounding); some hold only under conditions the paper leaves implicit, and the model
// measures by how much they fail otherwise. The binding comparison with holographic reduced
// representations (circular convolution in the same 32 coefficients) runs alongside.
#include <string>

#include "syn/models/models.hpp"
#include "syn/space/algebra.hpp"

// The agent program, in WGSL.
constexpr const char* fga_agent = R"wgsl(
  // Slots of the bank.
  let j = 0u;  let m = 1u;  let l = 2u;  let lEE = 3u;  let rP = 4u;  let rE = 5u;
  let rAG = 6u;  let rTH = 7u;  let ev = 8u;  let x = 9u;  let rot = 10u;  let rot2 = 11u;
  let t0 = 12u;  let t1 = 13u;  let t2 = 14u;  let t3 = 15u;  let t4 = 16u;  let t5 = 17u;
  let mo = 18u;  let w = 19u;
  // Entities: random points of the sphere of V_E = <A, B, C>.
  for (var k = 0u; k < 2u; k = k + 1u) {
    let z = 2.0 * U(k * 2u) - 1.0;
    let phi = 6.283185307 * U(k * 2u + 1u);
    let r = sqrt(max(0.0, 1.0 - z * z));
    mvVector(k, v5(0.0, r * cos(phi), r * sin(phi), z, 0.0));
  }
  // A relation: a bivector of V_E ∧ V_E (lEE) and with predicate content V_E ∧ V_P (l).
  mvZero(l);
  mvZero(lEE);
  let planes = array<u32, 6>(6u, 10u, 12u, 18u, 20u, 24u);  // AB AC BC Ae Be Ce
  for (var k = 0u; k < 6u; k = k + 1u) {
    let c = 2.0 * U(10u + k) - 1.0;
    mvSet(l, planes[k], c);
    if (k < 3u) { mvSet(lEE, planes[k], c); }
  }
  // A unary predicate: content in V_P, and a selectional part in V_E.
  mvZero(rP);
  mvSet(rP, 16u, 0.5 + U(20u));
  mvVector(rE, v5(0.0, 2.0 * U(21u) - 1.0, 2.0 * U(22u) - 1.0, 2.0 * U(23u) - 1.0, 0.0));
  // ---- Layer 1: application by contraction. s = j ⌋ (l ⌊ m) = <j l m>_0.
  mvMul(t0, l, m, 2u);
  mvMul(t1, j, t0, 1u);
  let s = mvGet(t1, 0u);
  mvMul(t2, j, l, 0u);
  mvMul(t3, t2, m, 0u);
  stat(0u, abs(s - mvGet(t3, 0u)));
  mvMul(t2, l, j, 2u);
  mvMul(t3, m, t2, 1u);
  stat(1u, abs(s + mvGet(t3, 0u)));  // s(John, Mary) + s(Mary, John)
  stat(2u, abs(s));
  stat(3u, mvSize(t0) - mvGradeSize(t0, 1u));
  stat(4u, mvSize(t1) - mvGradeSize(t1, 0u));
  // ---- Layer 2: event structure by the wedge, querying by contraction.
  mvVector(rAG, v5(1.0, 0.0, 0.0, 0.0, 0.0));
  mvVector(rTH, v5(0.0, 0.0, 0.0, 0.0, 1.0));
  mvMul(ev, rAG, j, 3u);
  mvMul(t0, rTH, m, 3u);
  mvAdd(ev, t0, 1.0);
  // r⁻¹ = r / r²: N⁻¹ = N, e⁻¹ = −e.
  mvMul(t1, rAG, ev, 1u);
  stat(5u, mvDistance(t1, j));
  mvCopy(t2, rTH);
  mvScale(t2, -1.0);
  mvMul(t1, t2, ev, 1u);
  stat(6u, mvDistance(t1, m));
  // The role of a filler, E ⌊ j⁻¹ (j unit): exact only when the fillers are orthogonal.
  mvMul(t1, ev, j, 2u);
  stat(7u, mvDistance(t1, rAG));
  let jm = mvScalar(j, m);
  mvCopy(mo, m);
  mvAdd(mo, j, -jm);
  mvScale(mo, 1.0 / max(mvSize(mo), 1e-6));
  mvMul(t3, rAG, j, 3u);
  mvMul(t0, rTH, mo, 3u);
  mvAdd(t3, t0, 1.0);
  mvMul(t1, t3, j, 2u);
  stat(8u, mvDistance(t1, rAG));
  // ---- Type checking: is a contraction with the wrong kind of argument zero?
  mvMul(t0, j, lEE, 1u);
  let base = max(mvSize(t0), 1e-6);
  mvMul(t1, rP, lEE, 1u);
  stat(9u, mvSize(t1) / base);    // a predicate purely in V_P against a relation in V_E ∧ V_E
  mvMul(t1, rE, lEE, 1u);
  stat(10u, mvSize(t1) / base);   // a predicate with a selectional part in V_E
  mvMul(t0, j, l, 1u);
  mvMul(t1, rP, l, 1u);
  stat(11u, mvSize(t1) / max(mvSize(t0), 1e-6));  // a relation with predicate content
  mvCopy(t2, rP);
  mvAdd(t2, rE, 1.0);
  mvMul(t1, t2, l, 3u);
  stat(12u, mvGradeSize(t1, 3u));  // run ∧ love: a trivector (ill-typed)
  mvMul(t1, j, l, 3u);
  stat(13u, mvGradeSize(t1, 3u));  // John ∧ love: also a trivector (well-typed)
  // ---- Layer 3: rotors. An elliptic plane (A∧B, squares to −1) and a hyperbolic one (A∧e).
  mvVector(x, v5(2.0 * U(30u) - 1.0, 2.0 * U(31u) - 1.0, 2.0 * U(32u) - 1.0, 2.0 * U(33u) - 1.0, 2.0 * U(34u) - 1.0));
  let theta = 6.283185307 * U(35u);
  for (var h = 0u; h < 2u; h = h + 1u) {
    mvPlaneRotor(rot, 1u, select(2u, 4u, h == 1u), theta);
    mvReverse(t0, rot);
    mvMul(t1, rot, t0, 0u);
    mvSet(t1, 0u, mvGet(t1, 0u) - 1.0);
    stat(14u + h, mvSize(t1));                       // R R̃ = 1
    mvSandwich(t2, rot, x, t3, t4);
    stat(16u + h, mvSize(t2) - mvGradeSize(t2, 1u)); // grade-preserving
    stat(18u + h, abs(mvScalar(t2, t2) - mvScalar(x, x)) / max(mvSize(t2) * mvSize(t2), 1.0));  // x² preserved
    mvReverse(t5, rot);
    mvSandwich(t1, t5, t2, t3, t4);
    stat(20u + h, mvDistance(t1, x) / max(mvSize(t2), 1.0));  // invertible: R̃ x' R = x
    mvPlaneRotor(rot2, 1u, select(2u, 4u, h == 1u), 6.283185307);
    mvSandwich(t1, rot2, x, t3, t4);
    stat(22u + h, mvDistance(t1, x));                 // a full turn: cyclic or not
    stat(24u + h, mvSize(t1) / mvSize(x));
  }
  // Composition: (R2 R1) x (R2 R1)~ = R2 (R1 x R̃1) R̃2.
  mvPlaneRotor(rot, 1u, 2u, theta);
  mvPlaneRotor(rot2, 3u, 4u, 0.5 * theta);
  mvSandwich(t2, rot, x, t3, t4);
  mvSandwich(t5, rot2, t2, t3, t4);
  mvMul(t0, rot2, rot, 0u);
  mvSandwich(t1, t0, x, t3, t4);
  stat(26u, mvDistance(t1, t5));
  // ---- Polar opposition: p± = ½(1 ± u) for a unit u of V_E (u² = +1), and for u = e (u² = −1).
  let z = 2.0 * U(40u) - 1.0;
  let phi = 6.283185307 * U(41u);
  let rr = sqrt(max(0.0, 1.0 - z * z));
  let uu = v5(0.0, rr * cos(phi), rr * sin(phi), z, 0.0);
  for (var h = 0u; h < 2u; h = h + 1u) {
    mvVector(t0, uu);
    if (h == 1u) { mvVector(t0, v5(0.0, 0.0, 0.0, 0.0, 1.0)); }
    mvZero(t1);
    mvSet(t1, 0u, 0.5);
    mvAdd(t1, t0, 0.5);   // p+
    mvZero(t2);
    mvSet(t2, 0u, 0.5);
    mvAdd(t2, t0, -0.5);  // p−
    mvMul(t3, t1, t1, 0u);
    mvAdd(t3, t1, -1.0);
    stat(27u + h * 3u, mvSize(t3));                  // p+² = p+
    mvMul(t3, t1, t2, 0u);
    stat(28u + h * 3u, mvSize(t3));                  // p+ p− = 0
    mvCopy(t3, t1);
    mvAdd(t3, t2, 1.0);
    mvSet(t3, 0u, mvGet(t3, 0u) - 1.0);
    stat(29u + h * 3u, mvSize(t3));                  // p+ + p− = 1
    if (h == 0u) {
      // Negation: −u·p+ (the paper's claim) and the reflection w p+ w⁻¹ in a w ⊥ u.
      mvMul(t3, t0, t1, 0u);
      mvScale(t3, -1.0);
      stat(33u, mvDistance(t3, t2));
      var wv = v5(0.0, -uu.x.z, uu.x.y, 0.0, 0.0);
      if (abs(uu.x.y) + abs(uu.x.z) < 1e-3) { wv = v5(0.0, 1.0, 0.0, 0.0, 0.0); }
      let wn = sqrt(wv.x.y * wv.x.y + wv.x.z * wv.x.z);
      mvVector(w, scale5(wv, 1.0 / wn));
      mvSandwich(t3, w, t1, t4, t5);
      stat(34u, mvDistance(t3, t2));
    }
  }
  // ---- A dot object: book = b_phys + b_form ∧ b_read + b_info; contraction recovers the qualia.
  mvCopy(t0, j);                       // b_phys
  mvCopy(t1, mo);                      // b_form (any unit of V_E)
  mvVector(t2, v5(0.0, 0.0, 0.0, 0.0, 1.0));  // b_read, in V_P
  mvMul(t3, t1, t2, 3u);
  mvAdd(t3, t0, 1.0);
  mvAdd(t3, t2, 0.5);                  // + b_info, also in V_P
  mvMul(t4, t1, t3, 1u);
  mvGrade(t5, t4, 1u);
  stat(35u, mvDistance(t5, t2));       // <b_form ⌋ book>_1 = b_read
  mvMul(t4, t2, t3, 1u);
  mvGrade(t5, t4, 1u);
  // e ⌋ book: the form back (plus e ⌋ b_phys = 0); e·e = −1 gives e ⌋ (f ∧ e) = f.
  stat(36u, mvDistance(t5, t1));
  // ---- Binding capacity: k role–filler pairs, recovered and cleaned up among eight fillers.
  // FGA: roles orthonormal in <N, e> for the first two; further roles r = cosh t N + sinh t e.
  // HRR: random 32-coefficient vectors bound by circular convolution, unbound by correlation.
  let voc = 20u;    // slots 20..27 the fillers
  let roles = 28u;  // slots 28..33 the roles
  let hev = 34u;
  for (var k = 0u; k < 8u; k = k + 1u) {
    let z2 = 2.0 * U(50u + k * 2u) - 1.0;
    let p2 = 6.283185307 * U(51u + k * 2u);
    let r2 = sqrt(max(0.0, 1.0 - z2 * z2));
    mvVector(voc + k, v5(0.0, r2 * cos(p2), r2 * sin(p2), z2, 0.0));
  }
  for (var n = 1u; n <= 6u; n = n + 1u) {
    mvZero(ev);
    for (var k = 0u; k < n; k = k + 1u) {
      var rv = v5(1.0, 0.0, 0.0, 0.0, 0.0);
      if (k == 1u) { rv = v5(0.0, 0.0, 0.0, 0.0, 1.0); }
      if (k >= 2u) {
        let tt = 2.0 * U(70u + n * 8u + k) - 1.0;
        rv = v5(cosh(tt), 0.0, 0.0, 0.0, sinh(tt));
      }
      mvVector(roles + k, rv);
      mvMul(t0, roles + k, voc + k, 3u);
      mvAdd(ev, t0, 1.0);
    }
    var right = 0.0;
    for (var k = 0u; k < n; k = k + 1u) {
      mvCopy(t1, roles + k);
      mvScale(t1, 1.0 / mvScalar(t1, t1));  // r⁻¹
      mvMul(t2, t1, ev, 1u);
      var best = 0u;
      var bestS = -1e9;
      for (var c = 0u; c < 8u; c = c + 1u) {
        let sc = mvScalar(t2, voc + c) / max(mvSize(t2), 1e-9);
        if (sc > bestS) { bestS = sc; best = c; }
      }
      right = right + select(0.0, 1.0, best == k);
    }
    stat(37u + n, right / f32(n));
  }
  // HRR: fillers and roles with 32 coefficients of variance 1/32 (Box–Muller).
  for (var k = 0u; k < 14u; k = k + 1u) {
    for (var i = 0u; i < 32u; i = i + 2u) {
      let u1 = U(200u + k * 32u + i);
      let u2 = U(201u + k * 32u + i);
      let g = sqrt(-2.0 * log(u1)) * 0.1767766953;
      mvSet(voc + k, i, g * cos(6.283185307 * u2));
      mvSet(voc + k, i + 1u, g * sin(6.283185307 * u2));
    }
  }
  for (var n = 1u; n <= 6u; n = n + 1u) {
    mvZero(hev);
    for (var k = 0u; k < n; k = k + 1u) {
      for (var o = 0u; o < 32u; o = o + 1u) {
        var acc = 0.0;
        for (var i = 0u; i < 32u; i = i + 1u) { acc = acc + mvGet(roles + k, i) * mvGet(voc + k, (o + 32u - i) % 32u); }
        mvSet(hev, o, mvGet(hev, o) + acc);
      }
    }
    var right = 0.0;
    var cosSum = 0.0;
    for (var k = 0u; k < n; k = k + 1u) {
      // Unbinding: correlation with the role, r*_i = r_(−i).
      for (var o = 0u; o < 32u; o = o + 1u) {
        var acc = 0.0;
        for (var i = 0u; i < 32u; i = i + 1u) { acc = acc + mvGet(roles + k, (32u - i) % 32u) * mvGet(hev, (o + 32u - i) % 32u); }
        mvSet(t2, o, acc);
      }
      var best = 0u;
      var bestS = -1e9;
      for (var c = 0u; c < 8u; c = c + 1u) {
        let sc = mvDot(t2, voc + c) / max(mvSize(t2) * mvSize(voc + c), 1e-9);
        if (sc > bestS) { bestS = sc; best = c; }
        if (c == k) { cosSum = cosSum + sc; }
      }
      right = right + select(0.0, 1.0, best == k);
    }
    stat(43u + n, right / f32(n));
    stat(49u + n, cosSum / f32(n));
  }
)wgsl";

namespace syn::models {

const kernel::Model* fgaAlgebraModel() {
  static const kernel::Model model = [] {
    kernel::Model m;
    m.name = "fgaAlgebra";
    m.title = "Functional geometric algebra: application, binding, type checks, rotors and polar idempotents, checked in Cl(4, 1)";
    m.maxStats = 56;
    m.agentSource = fga_agent;
    m.helpersWgsl = space::algebraWgsl(36);
    m.compile = [](const Value& spec) {
      const Value& c = spec.sub("fgaAlgebra");
      return std::vector<Plan>{makePlan("fgaAlgebra", static_cast<std::uint32_t>(c.integer("lexicons", 256)), 56, space::withSignTable({0.0}), {{0.0}},
                                        seedOf(spec))};
    };
    m.summarise = [](const Value&, const std::vector<Plan>& plans, const std::vector<Output>& out) {
      const Plan& p = plans[0];
      const Output& o = out[0];
      auto v = [&](std::uint32_t k) { return o.mean(p, 0, k); };
      const double exact = 1e-4;
      Value claims = Value::array();
      auto claim = [&](const char* section, const char* text, double measured, bool holds, const char* note) {
        claims.push(Value::object().set("section", section).set("claim", text).set("measured", measured).set("holds", holds).set("note", note));
      };
      claim("§5.3", "Application by contraction: j ⌋ (ℓ ⌊ m) is the scalar part of j ℓ m", v(0), v(0) < exact, "mean |j ⌋ (ℓ ⌊ m) − ⟨jℓm⟩₀|");
      claim("§5.3", "Argument order: John loves Mary ≠ Mary loves John", v(1) / v(2), v(1) / v(2) < exact,
            "|s(j, m) + s(m, j)| / |s|: a bivector relation is antisymmetric, so s(m, j) = −s(j, m) exactly — order is kept, and no "
            "bivector can make a relation symmetric (see the learning model)");
      claim("§2.4", "Grades are types: ℓ ⌊ m is a vector, j ⌋ (ℓ ⌊ m) a scalar", v(3) + v(4), v(3) + v(4) < exact, "size outside the expected grade");
      claim("§5.6", "Exact unbinding: r_AG⁻¹ ⌋ E = John, r_TH⁻¹ ⌋ E = Mary", v(5) + v(6), v(5) + v(6) < exact,
            "E = r_AG ∧ j + r_TH ∧ m, with r_AG = N, r_TH = e orthogonal to the entities");
      claim("§5.6", "Asymmetric querying: E ⌊ j⁻¹ gives the role John plays", v(7), v(7) < exact,
            "|E ⌊ j⁻¹ − r_AG|: the other binding leaks (j·m) r_TH, so it is exact only when the fillers are orthogonal");
      claim("§5.6", "… with orthogonal fillers", v(8), v(8) < exact, "E ⌊ j⁻¹ − r_AG with m ⊥ j");
      claim("§5.9", "Type check by nullity: run ⌋ love = 0 (predicate in V_P, relation in V_E ∧ V_E)", v(9), v(9) < exact, "|R_P ⌋ ℓ| / |j ⌋ ℓ|");
      claim("§5.9", "… with the selectional part §5.1 gives predicates (support in V_E)", v(10), v(10) < exact,
            "|R_E ⌋ ℓ| / |j ⌋ ℓ|: as large as a well-typed application — nullity needs the predicate to have no support in the relation's subspace");
      claim("§5.9", "… with the predicate content §5.2 gives relations (ℓ ∈ Cl²(V_E ⊕ V_P))", v(11), v(11) < exact,
            "|R_P ⌋ ℓ| / |j ⌋ ℓ|: not zero either");
      claim("§5.9", "The wedge confirms a type error: run ∧ love is a trivector", v(12), v(12) > 1e-2, "|⟨R ∧ ℓ⟩₃|");
      claim("§5.9", "… and only type errors give one", v(13), v(13) < exact,
            "|⟨j ∧ ℓ⟩₃|: John ∧ love, the well-typed binding of §5.5, is a trivector too — the grade alone does not separate them");
      claim("§2.5", "A rotor: R R̃ = 1 (elliptic and hyperbolic planes)", v(14) + v(15), v(14) + v(15) < exact, "");
      claim("§2.5", "Rotors preserve grade", v(16) + v(17), v(16) + v(17) < exact, "size of R x R̃ outside grade 1");
      claim("§2.5", "Rotors preserve the norm x²", v(18) + v(19), v(18) + v(19) < exact, "|⟨x′x′⟩ − ⟨xx⟩|");
      claim("§2.5", "Rotors are invertible: R̃ (R x R̃) R = x", v(20) + v(21), v(20) + v(21) < exact, "");
      claim("§2.5", "Rotors compose: (R₂R₁) x (R₂R₁)~ = R₂(R₁ x R̃₁)R̃₂", v(26), v(26) < exact, "");
      claim("§6.4", "Elliptic rotors are cyclic: a full turn returns", v(22), v(22) < exact, "plane A∧B (squares to −1)");
      claim("§6.4", "Hyperbolic rotors are not: a full turn in A∧e boosts", v(25), v(25) > 1.5,
            "size of x after a full turn / before (the plane contains e, e² = −1)");
      claim("§6.4", "Polar opposition: p± = ½(1 ± u) are complementary idempotents (u² = +1)", v(27) + v(28) + v(29), v(27) + v(28) + v(29) < exact,
            "p+² = p+, p+p− = 0, p+ + p− = 1");
      claim("§6.4", "… also on the negative axis u = e", v(30), v(30) < exact,
            "|p+² − p+| = ½: the paper asks for e² = +1; in Cl(4, 1) a polar axis must lie in ⟨N, A, B, C⟩, not in V_P = ⟨e⟩");
      claim("§6.4", "Negation as multiplication by −u swaps p+ and p−", v(33), v(33) < exact, "|−u p+ − p−| = 1: −u p+ = −p+");
      claim("§6.4", "… a reflection w p+ w⁻¹ in a w ⊥ u does", v(34), v(34) < exact, "");
      claim("§6.4", "A dot object: b_form ⌋ book recovers its telic quale b_read", v(35), v(35) < exact, "⟨b_form ⌋ book⟩₁ with book = b_phys + b_form ∧ b_read + b_info");
      claim("§6.4", "… and the purpose recovers the form: e ⌋ book", v(36), v(36) < exact, "⟨e ⌋ book⟩₁ − b_form");
      return Value::object()
          .set("claims", claims)
          .set("bindings", Value::Numbers({1, 2, 3, 4, 5, 6}))
          .set("fgaRecovery", Value::Numbers(o.range(p, 0, 38, 6)))
          .set("hrrRecovery", Value::Numbers(o.range(p, 0, 44, 6)))
          .set("hrrCosine", Value::Numbers(o.range(p, 0, 50, 6)))
          .set("antisymmetry", v(1) / v(2))
          .set("decomposition", std::vector<std::string>{"V_E = ⟨A, B, C⟩: entities (points of a 2-sphere), form", "V_P = ⟨e⟩: predicate content, purpose (e² = −1)",
                                                         "roles: r_AG = N, r_TH = e", "grades: 0 truth · 1 entities, predicates · 2 relations, bindings · 3 event frames"});
    };
    return m;
  }();
  return &model;
}

}  // namespace syn::models
