// The WGSL twin of syn/space/space.hpp: the same functions, the same names, in f32.
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

fn v5(n : f32, a : f32, b : f32, c : f32, e : f32) -> V5 { return V5(vec4<f32>(n, a, b, c), e); }
fn dot5(p : V5, q : V5) -> f32 { return dot(p.x, q.x) - p.e * q.e; }
fn add5(p : V5, q : V5) -> V5 { return V5(p.x + q.x, p.e + q.e); }
fn scale5(p : V5, s : f32) -> V5 { return V5(p.x * s, p.e * s); }
fn unit5(p : V5) -> V5 { return scale5(p, 1.0 / p.e); }
fn pole() -> V5 { return v5(1.0, 0.0, 0.0, 0.0, 1.0); }
fn antipode() -> V5 { return v5(-1.0, 0.0, 0.0, 0.0, 1.0); }
fn chanVec(channel : i32, s : f32) -> vec3<f32> {
  return vec3<f32>(select(0.0, s, channel == kA), select(0.0, s, channel == kB), select(0.0, s, channel == kC));
}
fn equator(channel : i32) -> V5 { let k = chanVec(channel, 1.0); return v5(0.0, k.x, k.y, k.z, 1.0); }
fn atChance(p : f32, channel : i32) -> V5 {
  let q = clamp(p, 1e-7, 1.0 - 1e-7);
  let k = chanVec(channel, 2.0 * sqrt(q * (1.0 - q)));
  return v5(2.0 * q - 1.0, k.x, k.y, k.z, 1.0);
}
fn chance(p : V5) -> f32 { return 0.5 * (1.0 + p.x.x / p.e); }
fn logit(p : V5) -> f32 { return log((p.e + p.x.x) / (p.e - p.x.x)); }
fn shift(p : V5, d : f32) -> V5 {
  let ch = cosh(0.5 * d);
  let sh = sinh(0.5 * d);
  return unit5(v5(ch * p.x.x + sh * p.e, p.x.y, p.x.z, p.x.w, sh * p.x.x + ch * p.e));
}
fn atLogit(x : f32, channel : i32) -> V5 { return shift(equator(channel), x); }
fn channelOf(p : V5) -> i32 {
  let a = abs(p.x.y);
  let b = abs(p.x.z);
  let c = abs(p.x.w);
  return select(select(kC, kB, b >= c), kA, a >= b && a >= c);
}
fn settle(p : V5, channel : i32) -> V5 { return atChance(chance(p), channel); }
fn mixture(p : V5, q : V5, t : f32) -> V5 { return add5(scale5(unit5(p), 1.0 - t), scale5(unit5(q), t)); }
fn raise(p : V5, t : f32) -> V5 { return settle(mixture(p, pole(), t), channelOf(p)); }
fn lower(p : V5, t : f32) -> V5 { return settle(mixture(p, antipode(), t), channelOf(p)); }
fn nudge(p : V5, d : f32) -> V5 {
  let c = chance(p);
  if (d >= 0.0) {
    if (c >= 1.0) { return p; }
    return raise(p, select(d / (1.0 - c), 1.0, d >= 1.0 - c));
  }
  if (c <= 0.0) { return p; }
  return lower(p, select(-d / c, 1.0, -d >= c));
}
fn blend(p : V5, q : V5, t : f32) -> V5 { return settle(mixture(p, q, t), channelOf(p)); }
fn rotate(p : V5, i : i32, j : i32, alpha : f32) -> V5 {
  var r = p;
  let c = cos(alpha);
  let s = sin(alpha);
  r.x[i] = c * p.x[i] - s * p.x[j];
  r.x[j] = s * p.x[i] + c * p.x[j];
  return r;
}
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
fn similarity(p : V5, q : V5) -> f32 { return 1.0 + 0.5 * dot5(unit5(p), unit5(q)); }
fn cap(centre : V5, rho : f32) -> V5 { let c = unit5(centre); return V5(c.x, cos(rho)); }
fn chanceCap(p : f32) -> V5 { return v5(1.0, 0.0, 0.0, 0.0, 2.0 * p - 1.0); }
fn side(p : V5, s : V5) -> f32 { return dot5(unit5(p), s); }
fn inside(p : V5, s : V5) -> bool { return side(p, s) >= 0.0; }
fn lift(u1 : f32, u2 : f32, u3 : f32, scale : f32) -> V5 {
  let u = vec3<f32>(u1, u2, u3) / scale;
  let r2 = dot(u, u);
  let k = 1.0 / (1.0 + r2);
  return v5((1.0 - r2) * k, 2.0 * u.x * k, 2.0 * u.y * k, 2.0 * u.z * k, 1.0);
}
fn pointOf(p : V5) -> V5 { return V5(normalize(p.x), 1.0); }
fn logisticOf(u : f32) -> f32 { return log(u / (1.0 - u)); }
fn sampleLevel(u : f32, mean : f32, sd : f32, channel : i32) -> V5 {
  return atLogit(mean + sd * kLogisticUnit * logisticOf(u), channel);
}
fn sampleShare(u : f32, m : f32, sd : f32, channel : i32) -> V5 {
  let mm = clamp(m, 0.02, 0.98);
  let spread = sd / (mm * (1.0 - mm));
  return atLogit(log(mm / (1.0 - mm)) + spread * kLogisticUnit * logisticOf(u), channel);
}
)wgsl";
  return source;
}

}  // namespace syn::space
