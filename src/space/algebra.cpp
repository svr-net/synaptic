// The algebra of the space, in WGSL: multivectors of Cl(4, 1) (see syn/space/algebra.hpp).
#include "syn/space/algebra.hpp"

#include "syn/core/multivector.hpp"

namespace syn::space {

std::vector<double> signTable() {
  std::vector<double> t(kSignTable);
  for (unsigned a = 0; a < kBlades; ++a)
    for (unsigned b = 0; b < kBlades; ++b) t[a * kBlades + b] = Multivector::bladeSign(a, b, 4);
  return t;
}

std::vector<double> withSignTable(const std::vector<double>& constants) {
  std::vector<double> p = signTable();
  p.insert(p.end(), constants.begin(), constants.end());
  return p;
}

std::string algebraWgsl(unsigned slots) {
  std::string s = "\nconst MVS : u32 = " + std::to_string(slots) + "u;\nvar<private> MV : array<f32, " + std::to_string(slots * kBlades) + ">;\n";
  s += R"wgsl(
// ---- the algebra of the space: multivectors of Cl(4, 1) in the slots of the bank MV ----
// Blade m: bit 0 N, 1 A, 2 B, 3 C, 4 e. e_a e_b = bladeSign(a, b) e_(a ^ b).
fn Q(i : u32) -> f32 { return P(1024u + i); }
fn bladeSign(a : u32, b : u32) -> f32 { return P(a * 32u + b); }
fn gradeOf(m : u32) -> u32 { return (m & 1u) + ((m >> 1u) & 1u) + ((m >> 2u) & 1u) + ((m >> 3u) & 1u) + ((m >> 4u) & 1u); }
fn mvGet(s : u32, b : u32) -> f32 { return MV[s * 32u + b]; }
fn mvSet(s : u32, b : u32, v : f32) { MV[s * 32u + b] = v; }
fn mvZero(d : u32) { for (var i = 0u; i < 32u; i = i + 1u) { MV[d * 32u + i] = 0.0; } }
fn mvCopy(d : u32, s : u32) { for (var i = 0u; i < 32u; i = i + 1u) { MV[d * 32u + i] = MV[s * 32u + i]; } }
fn mvScale(d : u32, k : f32) { for (var i = 0u; i < 32u; i = i + 1u) { MV[d * 32u + i] = k * MV[d * 32u + i]; } }
// d ← d + k·s
fn mvAdd(d : u32, s : u32, k : f32) { for (var i = 0u; i < 32u; i = i + 1u) { MV[d * 32u + i] = MV[d * 32u + i] + k * MV[s * 32u + i]; } }
// A vector of the space as a multivector.
fn mvVector(d : u32, p : V5) {
  mvZero(d);
  MV[d * 32u + 1u] = p.x.x;
  MV[d * 32u + 2u] = p.x.y;
  MV[d * 32u + 4u] = p.x.z;
  MV[d * 32u + 8u] = p.x.w;
  MV[d * 32u + 16u] = p.e;
}
// d ← a ∘ b (d different from a and b): op 0 the geometric product, 1 the left contraction a ⌋ b,
// 2 the right contraction a ⌊ b, 3 the wedge a ∧ b. On blades: a ⌋ b = ab when a ⊆ b,
// a ⌊ b = ab when b ⊆ a, a ∧ b = ab when a ∩ b = ∅, and 0 otherwise.
fn mvMul(d : u32, a : u32, b : u32, op : u32) {
  mvZero(d);
  for (var i = 0u; i < 32u; i = i + 1u) {
    let x = MV[a * 32u + i];
    if (x == 0.0) { continue; }
    for (var j = 0u; j < 32u; j = j + 1u) {
      let y = MV[b * 32u + j];
      if (y == 0.0) { continue; }
      let both = i & j;
      var keep = true;
      if (op == 1u) { keep = both == i; }
      if (op == 2u) { keep = both == j; }
      if (op == 3u) { keep = both == 0u; }
      if (keep) { MV[d * 32u + (i ^ j)] = MV[d * 32u + (i ^ j)] + bladeSign(i, j) * x * y; }
    }
  }
}
// ⟨a b⟩₀, the scalar part of the geometric product.
fn mvScalar(a : u32, b : u32) -> f32 {
  var s = 0.0;
  for (var i = 0u; i < 32u; i = i + 1u) { s = s + MV[a * 32u + i] * MV[b * 32u + i] * bladeSign(i, i); }
  return s;
}
// The Euclidean dot product of the coefficients (no metric): for flat vectors of 32 numbers.
fn mvDot(a : u32, b : u32) -> f32 {
  var s = 0.0;
  for (var i = 0u; i < 32u; i = i + 1u) { s = s + MV[a * 32u + i] * MV[b * 32u + i]; }
  return s;
}
// d ← the reverse of s: grades 2 and 3 change sign.
fn mvReverse(d : u32, s : u32) {
  for (var i = 0u; i < 32u; i = i + 1u) {
    let g = gradeOf(i);
    MV[d * 32u + i] = select(1.0, -1.0, g == 2u || g == 3u) * MV[s * 32u + i];
  }
}
// d ← ⟨s⟩_k
fn mvGrade(d : u32, s : u32, k : u32) {
  for (var i = 0u; i < 32u; i = i + 1u) { MV[d * 32u + i] = select(0.0, MV[s * 32u + i], gradeOf(i) == k); }
}
// The size of grade k of s, and of all of s (Euclidean, on the coefficients).
fn mvGradeSize(s : u32, k : u32) -> f32 {
  var t = 0.0;
  for (var i = 0u; i < 32u; i = i + 1u) { if (gradeOf(i) == k) { t = t + MV[s * 32u + i] * MV[s * 32u + i]; } }
  return sqrt(t);
}
fn mvSize(s : u32) -> f32 {
  var t = 0.0;
  for (var i = 0u; i < 32u; i = i + 1u) { t = t + MV[s * 32u + i] * MV[s * 32u + i]; }
  return sqrt(t);
}
// The size of a − b.
fn mvDistance(a : u32, b : u32) -> f32 {
  var t = 0.0;
  for (var i = 0u; i < 32u; i = i + 1u) { let x = MV[a * 32u + i] - MV[b * 32u + i]; t = t + x * x; }
  return sqrt(t);
}
// The rotor exp(−θ/2 · e_i e_j) of the coordinate plane (i, j) (bit indices, i < j): elliptic
// (cos, sin) when the plane squares to −1, hyperbolic (cosh, sinh) when it contains e.
fn mvPlaneRotor(d : u32, i : u32, j : u32, theta : f32) {
  mvZero(d);
  let blade = (1u << i) | (1u << j);
  // Normalised so that R R̃ = 1 exactly, whatever the precision of the device's cos and sin.
  if (j == 4u) {
    let c = cosh(0.5 * theta);
    let s = sinh(0.5 * theta);
    let n = sqrt(max(c * c - s * s, 1e-12));
    MV[d * 32u] = c / n;
    MV[d * 32u + blade] = -s / n;
  } else {
    let c = cos(0.5 * theta);
    let s = sin(0.5 * theta);
    let n = sqrt(c * c + s * s);
    MV[d * 32u] = c / n;
    MV[d * 32u + blade] = -s / n;
  }
}
// r ← r / √⟨r r̃⟩₀ (t a scratch slot): back onto the rotors after many products.
fn mvNormaliseRotor(r : u32, t : u32) {
  mvReverse(t, r);
  let n = mvScalar(r, t);
  mvScale(r, 1.0 / sqrt(max(abs(n), 1e-12)));
}
// d ← r x r̃, with t and u scratch slots (all five different).
fn mvSandwich(d : u32, r : u32, x : u32, t : u32, u : u32) {
  mvMul(t, r, x, 0u);
  mvReverse(u, r);
  mvMul(d, t, u, 0u);
}
)wgsl";
  return s;
}

}  // namespace syn::space
