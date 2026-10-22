// Carried verbatim from svr-net/compute@c8b8594141f292c3a638e6735c0ff319d36ea76f (include/compute/algebra_cga.hpp).
// Refresh with tools/sync_compute.sh; do not edit by hand.
/******************************************************************************************
    compute/algebra_cga.hpp -- the whole algebra Cl(4,1) of the elliptic conformal model of S³

    Every grade of the space, not only its null vectors: scalars, vectors, bivectors and up, with the
    geometric product, the left and right contractions, the wedge, the reverse and grade projection.
    A multivector has 32 coefficients, one per basis blade; a blade is a bit mask over the basis vectors
    (bit 0 N, 1 A, 2 B, 3 C -- the four Euclidean directions S³ sits in, squaring to +1 -- and bit 4 e,
    e² = −1). e_a e_b = bladeSign(a, b) e_(a ⊕ b).

      * Mv, product, wedge, left, right, reverse, grade, scalar, sandwich     the algebra on the CPU, in
        double: the reference every kernel is checked against.
      * point, cap, side, angle, rotorTowards, dilation, lift, project, centre    the elliptic model of S³
        in it: a point X = x + e (x on S³) is null, X·Y = cos θ − 1, a cap S = ĉ + cos ρ·e holds the
        points with X·S ≥ 0, rotors are the isometries and dilations exp(β/2·n e) the conformal maps
        fixing ±n.
      * algebraWgsl(slots)      the same operations in WGSL, on a private bank MV of `slots` multivectors
        (a kernel cannot pass arrays to functions portably). The products read their signs through
        `bladeSign(a, b)`, which the including kernel defines: kBladeSignWgsl computes it from the bits, or
        a kernel with the table among its constants (signTable()) reads it from there.
******************************************************************************************/

#pragma once

#include <array>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

namespace compute::algebra {

inline constexpr unsigned kBlades = 32;
inline constexpr unsigned kSignTable = kBlades * kBlades;
inline constexpr unsigned kE = 4;   // the index of e, the vector with e² = −1

constexpr int gradeOf(std::uint32_t m) {
    int g = 0;
    for (; m; m &= m - 1) ++g;
    return g;
}

/// The sign of e_a e_b: each vector of b passes every vector of a with a higher index, and the vectors
/// common to both contract to their squares (e² = −1).
constexpr double bladeSign(std::uint32_t a, std::uint32_t b) {
    int swaps = 0;
    for (std::uint32_t x = a >> 1; x; x >>= 1) swaps += gradeOf(x & b);
    double s = (swaps & 1) ? -1.0 : 1.0;
    if ((a & b) >> kE & 1u) s = -s;
    return s;
}

/// The signs row-major (a · 32 + b): the table a kernel can carry among its constants.
inline std::vector<double> signTable() {
    std::vector<double> t(kSignTable);
    for (std::uint32_t a = 0; a < kBlades; ++a)
        for (std::uint32_t b = 0; b < kBlades; ++b) t[a * kBlades + b] = bladeSign(a, b);
    return t;
}

/// A multivector of Cl(4,1): one coefficient per blade.
struct Mv {
    std::array<double, kBlades> c{};

    double& operator[](std::uint32_t m) { return c[m]; }
    double operator[](std::uint32_t m) const { return c[m]; }

    static Mv scalar(double s) { Mv r; r[0] = s; return r; }
    static Mv basis(unsigned i) { Mv r; r[1u << i] = 1.0; return r; }
    /// n N + a A + b B + c C + e·e
    static Mv vector(double n, double a, double b, double cc, double e) {
        Mv r;
        r[1] = n, r[2] = a, r[4] = b, r[8] = cc, r[16] = e;
        return r;
    }

    Mv operator+(const Mv& o) const { Mv r = *this; for (unsigned i = 0; i < kBlades; ++i) r.c[i] += o.c[i]; return r; }
    Mv operator-(const Mv& o) const { Mv r = *this; for (unsigned i = 0; i < kBlades; ++i) r.c[i] -= o.c[i]; return r; }
    Mv operator*(double k) const { Mv r = *this; for (auto& x : r.c) x *= k; return r; }
    Mv operator/(double k) const { return *this * (1.0 / k); }
    double maxAbs() const { double m = 0; for (double x : c) m = std::fmax(m, std::fabs(x)); return m; }
};

namespace detail {
// op 0 the geometric product, 1 the left contraction (a ⊆ b), 2 the right (b ⊆ a), 3 the wedge (a ∩ b = ∅).
inline Mv multiply(const Mv& x, const Mv& y, int op) {
    Mv z;
    for (std::uint32_t a = 0; a < kBlades; ++a) {
        if (x[a] == 0.0) continue;
        for (std::uint32_t b = 0; b < kBlades; ++b) {
            if (y[b] == 0.0) continue;
            const std::uint32_t both = a & b;
            if ((op == 1 && both != a) || (op == 2 && both != b) || (op == 3 && both != 0)) continue;
            z[a ^ b] += bladeSign(a, b) * x[a] * y[b];
        }
    }
    return z;
}
}  // namespace detail

inline Mv product(const Mv& a, const Mv& b) { return detail::multiply(a, b, 0); }
inline Mv left(const Mv& a, const Mv& b) { return detail::multiply(a, b, 1); }
inline Mv right(const Mv& a, const Mv& b) { return detail::multiply(a, b, 2); }
inline Mv wedge(const Mv& a, const Mv& b) { return detail::multiply(a, b, 3); }

/// Grades 2 and 3 change sign.
inline Mv reverse(const Mv& a) {
    Mv r = a;
    for (std::uint32_t m = 0; m < kBlades; ++m) {
        const int k = gradeOf(m);
        if ((k * (k - 1) / 2) & 1) r[m] = -r[m];
    }
    return r;
}
inline Mv grade(const Mv& a, int k) {
    Mv r;
    for (std::uint32_t m = 0; m < kBlades; ++m)
        if (gradeOf(m) == k) r[m] = a[m];
    return r;
}
/// ⟨a b⟩₀: each blade pairs only with itself.
inline double scalar(const Mv& a, const Mv& b) {
    double s = 0.0;
    for (std::uint32_t m = 0; m < kBlades; ++m) s += bladeSign(m, m) * a[m] * b[m];
    return s;
}
/// V X Ṽ
inline Mv sandwich(const Mv& v, const Mv& x) { return product(product(v, x), reverse(v)); }

// ---- the elliptic model of S³ ----

inline Mv e() { return Mv::basis(kE); }
/// The weight λ = −X·e of a vector (1 for a normalised point).
inline double weight(const Mv& x) { return x[16]; }
inline Mv normalise(const Mv& x) { return x / weight(x); }
/// The normalised point x + e of a direction (any length).
inline Mv point(double n, double a, double b, double c) {
    const double r = std::sqrt(n * n + a * a + b * b + c * c);
    return Mv::vector(n / r, a / r, b / r, c / r, 1.0);
}
inline Mv point(const std::array<double, 4>& d) { return point(d[0], d[1], d[2], d[3]); }
inline bool isPoint(const Mv& x, double tol = 1e-9) {
    return (x - grade(x, 1)).maxAbs() <= tol && std::fabs(scalar(x, x)) <= tol * std::fmax(1.0, x.maxAbs() * x.maxAbs());
}
/// The geodesic angle between two points: atan2(|x ∧ y|, x·y) of their directions (accurate near 0 and π).
inline double angle(const Mv& x, const Mv& y) {
    const Mv a = normalise(x) - e(), b = normalise(y) - e();
    const Mv w = wedge(a, b);
    return std::atan2(std::sqrt(std::fabs(scalar(w, reverse(w)))), scalar(a, b));
}
/// The cap ĉ + cos ρ·e: the points within angle ρ of c.
inline Mv cap(const Mv& centre, double rho) {
    Mv s = normalise(centre);
    s[16] = std::cos(rho);
    return s;
}
/// X·S = cos θ − cos ρ for a normalised point: > 0 inside the cap.
inline double side(const Mv& x, const Mv& s) { return scalar(normalise(x), s); }
/// The rotor exp(−tθ/2 · B̂) moving a towards b along their geodesic by the fraction t.
inline Mv rotorTowards(const Mv& a, const Mv& b, double t) {
    const Mv x = normalise(a) - e(), y = normalise(b) - e();
    const Mv plane = wedge(x, y);
    const double s2 = scalar(plane, reverse(plane));
    if (s2 <= 1e-30) return Mv::scalar(1.0);
    const double theta = std::atan2(std::sqrt(s2), scalar(x, y));
    // y x = x·y − x∧y: the rotor e^{−φ/2 B̂} with B̂ = x∧y/|x∧y| turns x towards y by φ
    return Mv::scalar(std::cos(t * theta / 2)) - plane / std::sqrt(s2) * std::sin(t * theta / 2);
}
/// The dilation exp(β/2 · n e) fixing ±n: tan(φ/2) ↦ e^β tan(φ/2), φ the angle from n.
inline Mv dilation(const std::array<double, 4>& pole, double beta) {
    const Mv n = point(pole) - e();
    return Mv::scalar(std::cosh(beta / 2)) + product(n, e()) * std::sinh(beta / 2);
}
/// V X Ṽ, normalised to a point.
inline Mv move(const Mv& v, const Mv& x) { return normalise(grade(sandwich(v, x), 1)); }
/// The centre of points: the direction of their sum.
inline Mv centre(const std::vector<Mv>& points) {
    Mv s;
    for (const auto& p : points) s = s + normalise(p);
    return point(s[1], s[2], s[4], s[8]);
}
/// Inverse stereographic projection of u / scale onto S³, with N the pole.
inline Mv lift(double u1, double u2, double u3, double scale = 1.0) {
    u1 /= scale, u2 /= scale, u3 /= scale;
    const double r2 = u1 * u1 + u2 * u2 + u3 * u3, k = 1.0 / (1.0 + r2);
    return Mv::vector((1.0 - r2) * k, 2.0 * u1 * k, 2.0 * u2 * k, 2.0 * u3 * k, 1.0);
}
/// Back to R³: the stereographic projection from −N.
inline std::array<double, 3> project(const Mv& x, double scale = 1.0) {
    const Mv p = normalise(x);
    const double k = scale / (1.0 + p[1]);
    return {p[2] * k, p[4] * k, p[8] * k};
}

// ---- the same algebra in WGSL ----

/// `bladeSign(a, b)` from the bits, for a kernel that does not carry the table.
inline constexpr const char* kBladeSignWgsl = R"WGSL(
fn bladeSign(a : u32, b : u32) -> f32 {
  var swaps = 0u;
  for (var x = a >> 1u; x != 0u; x = x >> 1u) {
    for (var m = x & b; m != 0u; m = m & (m - 1u)) { swaps = swaps + 1u; }
  }
  var s = select(1.0, -1.0, (swaps & 1u) == 1u);
  if (((a & b) & 16u) != 0u) { s = -s; }
  return s;
}
)WGSL";

inline constexpr const char* kAlgebraWgsl = R"WGSL(
// ---- the algebra Cl(4, 1) in the slots of the bank MV (compute/algebra_cga.hpp) ----
// Blade m: bit 0 N, 1 A, 2 B, 3 C, 4 e. e_a e_b = bladeSign(a, b) e_(a ^ b).
fn gradeOf(m : u32) -> u32 { return (m & 1u) + ((m >> 1u) & 1u) + ((m >> 2u) & 1u) + ((m >> 3u) & 1u) + ((m >> 4u) & 1u); }
fn mvGet(s : u32, b : u32) -> f32 { return MV[s * 32u + b]; }
fn mvSet(s : u32, b : u32, v : f32) { MV[s * 32u + b] = v; }
fn mvZero(d : u32) { for (var i = 0u; i < 32u; i = i + 1u) { MV[d * 32u + i] = 0.0; } }
fn mvCopy(d : u32, s : u32) { for (var i = 0u; i < 32u; i = i + 1u) { MV[d * 32u + i] = MV[s * 32u + i]; } }
fn mvScale(d : u32, k : f32) { for (var i = 0u; i < 32u; i = i + 1u) { MV[d * 32u + i] = k * MV[d * 32u + i]; } }
// d ← d + k·s
fn mvAdd(d : u32, s : u32, k : f32) { for (var i = 0u; i < 32u; i = i + 1u) { MV[d * 32u + i] = MV[d * 32u + i] + k * MV[s * 32u + i]; } }
// d ← n N + a A + b B + c C + e·e
fn mvVector5(d : u32, n : f32, a : f32, b : f32, c : f32, e : f32) {
  mvZero(d);
  MV[d * 32u + 1u] = n;
  MV[d * 32u + 2u] = a;
  MV[d * 32u + 4u] = b;
  MV[d * 32u + 8u] = c;
  MV[d * 32u + 16u] = e;
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
)WGSL";

/// The bank MV of `slots` multivectors and the operations on it (the kernel defines bladeSign).
inline std::string algebraWgsl(unsigned slots) {
    return "\nconst MVS : u32 = " + std::to_string(slots) + "u;\nvar<private> MV : array<f32, " +
           std::to_string(slots * kBlades) + ">;\n" + kAlgebraWgsl;
}

}  // namespace compute::algebra
