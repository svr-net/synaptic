// The space, in WGSL: every kernel of the library includes it (see syn/space/space.hpp).
#include "syn/space/space.hpp"

namespace syn::space {

const std::string& spaceWgsl() {
  static const std::string source = R"wgsl(
// ---- the space: pure elliptic CGA Cl(4, 1); x = (N, A, B, C), e with e² = −1 ----
struct V5 { x : vec4<f32>, e : f32, };
const kN : i32 = 0;
const kA : i32 = 1;
const kB : i32 = 2;
const kC : i32 = 3;
const kLogisticUnit : f32 = 0.5513288954217921;

// A vector of Cl(4, 1): x = (N, A, B, C), e the coefficient of e (e² = −1).
fn v5(n : f32, a : f32, b : f32, c : f32, e : f32) -> V5 { return V5(vec4<f32>(n, a, b, c), e); }
// Inner product: N² = A² = B² = C² = +1, e² = −1.
fn dot5(p : V5, q : V5) -> f32 { return dot(p.x, q.x) - p.e * q.e; }
fn add5(p : V5, q : V5) -> V5 { return V5(p.x + q.x, p.e + q.e); }
fn scale5(p : V5, s : f32) -> V5 { return V5(p.x * s, p.e * s); }
// X / λ with λ = −X·e the weight (1 for a normalised point).
fn unit5(p : V5) -> V5 { return scale5(p, 1.0 / p.e); }
// The learner L = N + e (chance 1) and its antipode −N + e (chance 0).
fn pole() -> V5 { return v5(1.0, 0.0, 0.0, 0.0, 1.0); }
fn antipode() -> V5 { return v5(-1.0, 0.0, 0.0, 0.0, 1.0); }
fn chanVec(channel : i32, s : f32) -> vec3<f32> {
  return vec3<f32>(select(0.0, s, channel == kA), select(0.0, s, channel == kB), select(0.0, s, channel == kC));
}
// The point of chance ½ in a channel: K + e.
fn equator(channel : i32) -> V5 { let k = chanVec(channel, 1.0); return v5(0.0, k.x, k.y, k.z, 1.0); }
// The point of chance p in a channel: (2p − 1)·N + 2√(p(1 − p))·K + e.
fn atChance(p : f32, channel : i32) -> V5 {
  let q = clamp(p, 1e-7, 1.0 - 1e-7);
  let k = chanVec(channel, 2.0 * sqrt(q * (1.0 - q)));
  return v5(2.0 * q - 1.0, k.x, k.y, k.z, 1.0);
}
// chance(X) = 1 + ½·X̂·L = ½(1 + X_N / X_e).
fn chance(p : V5) -> f32 { return 0.5 * (1.0 + p.x.x / p.e); }
// logit(X) = ln(chance / (1 − chance)) = ln((X_e + X_N) / (X_e − X_N)).
fn logit(p : V5) -> f32 { return log((p.e + p.x.x) / (p.e - p.x.x)); }
// The dilation D = exp(−d/4 · N e), normalised: adds d to the logit.
fn shift(p : V5, d : f32) -> V5 {
  let ch = cosh(0.5 * d);
  let sh = sinh(0.5 * d);
  return unit5(v5(ch * p.x.x + sh * p.e, p.x.y, p.x.z, p.x.w, sh * p.x.x + ch * p.e));
}
// The point of logit x in a channel: the dilation of the channel's equator point.
fn atLogit(x : f32, channel : i32) -> V5 { return shift(equator(channel), x); }
// The channel a point lies in (its largest content component; the cognitive one if none).
fn channelOf(p : V5) -> i32 {
  let a = abs(p.x.y);
  let b = abs(p.x.z);
  let c = abs(p.x.w);
  return select(select(kC, kB, b >= c), kA, a >= b && a >= c);
}
// Back onto the circle (N, K): the null point of the same chance in channel K.
fn settle(p : V5, channel : i32) -> V5 { return atChance(chance(p), channel); }
// The mixture (1 − t)·X̂ + t·Ŷ of two normalised vectors; its chance is the same mixture.
fn mixture(p : V5, q : V5, t : f32) -> V5 { return add5(scale5(unit5(p), 1.0 - t), scale5(unit5(q), t)); }
// Mixture towards the learner, settled: chance ↦ chance + t·(1 − chance).
fn raise(p : V5, t : f32) -> V5 { return settle(mixture(p, pole(), t), channelOf(p)); }
// Mixture towards the antipode, settled: chance ↦ chance·(1 − t).
fn lower(p : V5, t : f32) -> V5 { return settle(mixture(p, antipode(), t), channelOf(p)); }
// Additive change of the reading by d (clamped to [0, 1]) as a raise or a lower.
fn nudge(p : V5, d : f32) -> V5 {
  let c = chance(p);
  if (d >= 0.0) {
    if (c >= 1.0) { return p; }
    return raise(p, select(d / (1.0 - c), 1.0, d >= 1.0 - c));
  }
  if (c <= 0.0) { return p; }
  return lower(p, select(-d / c, 1.0, -d >= c));
}
// The mixture of X and Y with weight t on Y, settled in X's channel.
fn blend(p : V5, q : V5, t : f32) -> V5 { return settle(mixture(p, q, t), channelOf(p)); }
// Rotor in the coordinate plane (i, j) of the Euclidean directions: i turns towards j by α.
fn rotate(p : V5, i : i32, j : i32, alpha : f32) -> V5 {
  var r = p;
  let c = cos(alpha);
  let s = sin(alpha);
  r.x[i] = c * p.x[i] - s * p.x[j];
  r.x[j] = s * p.x[i] + c * p.x[j];
  return r;
}
// The rotor exp(−tθ/2 · B̂) that turns X a fraction t along the geodesic to Y.
fn towards(p : V5, q : V5, t : f32) -> V5 {
  let a = unit5(p);
  let b = unit5(q);
  let c = clamp(dot(a.x, b.x), -1.0, 1.0);
  let theta = acos(c);
  let s = sin(theta);
  if (s < 1e-6) { return a; }
  let k0 = cos(t * theta) - c * sin(t * theta) / s;
  let k1 = sin(t * theta) / s;
  return V5(k0 * a.x + k1 * b.x, 1.0);
}
// 1 + ½·X̂·Ŷ = cos²(θ/2): generalisation, fit to a schema.
fn similarity(p : V5, q : V5) -> f32 { return 1.0 + 0.5 * dot5(unit5(p), unit5(q)); }
// The cap S = ĉ + cos ρ·e: the points within angle ρ of c.
fn cap(centre : V5, rho : f32) -> V5 { let c = unit5(centre); return V5(c.x, cos(rho)); }
// The cap of the points whose chance is at least p: a random event is inside(X, chanceCap(u)).
fn chanceCap(p : f32) -> V5 { return v5(1.0, 0.0, 0.0, 0.0, 2.0 * p - 1.0); }
// X̂·S = cos θ − cos ρ: > 0 inside the cap, 0 on its boundary, < 0 outside.
fn side(p : V5, s : V5) -> f32 { return dot5(unit5(p), s); }
fn inside(p : V5, s : V5) -> bool { return side(p, s) >= 0.0; }
// Inverse stereographic projection of u / scale onto S³ (with N the pole).
fn lift(u1 : f32, u2 : f32, u3 : f32, scale : f32) -> V5 {
  let u = vec3<f32>(u1, u2, u3) / scale;
  let r2 = dot(u, u);
  let k = 1.0 / (1.0 + r2);
  return v5((1.0 - r2) * k, 2.0 * u.x * k, 2.0 * u.y * k, 2.0 * u.z * k, 1.0);
}
// The centre of a set of points: the direction of the sum of their null vectors.
fn pointOf(p : V5) -> V5 { return V5(normalize(p.x), 1.0); }
// The logit of a uniform u: the space's own distribution (sd π/√3).
fn logisticOf(u : f32) -> f32 { return log(u / (1.0 - u)); }
// A random point whose logit is logistic around `mean` with sd `sd`.
fn sampleLevel(u : f32, mean : f32, sd : f32, channel : i32) -> V5 {
  return atLogit(mean + sd * kLogisticUnit * logisticOf(u), channel);
}
// A random point whose chance is a share around m with sd `sd` (logistic on the logit scale).
fn sampleShare(u : f32, m : f32, sd : f32, channel : i32) -> V5 {
  let mm = clamp(m, 0.02, 0.98);
  let spread = sd / (mm * (1.0 - mm));
  return atLogit(log(mm / (1.0 - mm)) + spread * kLogisticUnit * logisticOf(u), channel);
}
)wgsl";
  return source;
}

}  // namespace syn::space
