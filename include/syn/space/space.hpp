#pragma once

// The space: one pure elliptic conformal geometric algebra in which every model of the library
// lives, on the CPU (this header, in float or double) and on the GPU (the WGSL twin returned by
// spaceWgsl(), the same functions under the same names).
//
// The algebra is Cl(4, 1), the conformal model of the 3-sphere S³: four Euclidean directions
//   N  the pole: the learner (or, in general, the agent the model is about)
//   A  the cognitive channel   (knowledge, skill, memory, schemas, levels of Bloom's taxonomy)
//   B  the affective channel   (anxiety, efficacy, interest, value, mindset, needs)
//   C  the social channel      (cultural capital, belonging, peers, community, raters)
// and e with e² = −1. There is no origin and no point at infinity: a point is a null vector
// X = x + e with x ∈ S³, and the geometry is that of angles on the sphere.
//
// Scalars are the miniature basis of the space. A probability, a share or a level is not stored
// as a number but as a point; the number is a reading of the point against the pole:
//   chance(X) = 1 + ½·X·L = cos²(φ/2)        L = N + e the learner, φ the angle of X from N
//   logit(X)  = ln(chance / (1 − chance)) = −2·ln tan(φ/2)
// so the point of chance p in channel K is X = (2p − 1)·N + 2√(p(1 − p))·K + e. Everything a
// model does to a quantity is an operation of the algebra on its point:
//   shift(X, d)     the dilation versor D = exp(−d/4 · N e): tan(φ/2) ↦ e^{−d/2} tan(φ/2), i.e.
//                   logit ↦ logit + d. Effects that add on the logit scale (a lever, a scaffold,
//                   a bias, a threshold) are dilations about the pole, and they compose by adding.
//   raise(X, t)     the mixture (1 − t)·X + t·L, settled back onto the circle of its channel:
//   lower(X, t)     chance ↦ chance + t·(1 − chance), or chance·(1 − t) towards the antipode. The
//                   reading is linear in mixtures, so learning-rate updates (Rescorla–Wagner,
//                   efficacy, habit) are mixtures.
//   towards(X,Y,t)  the rotor exp(−tθ/2 · B̂) that turns X along the geodesic to Y.
//   rotate(X,i,j,α) a rotor in a coordinate plane: moves content between channels.
//   similarity(X,Y) 1 + ½·X·Y = cos²(θ/2) between two points: generalisation, fit to a schema.
//   inside(X, S)    X·S ≥ 0 for the cap S = c + cos ρ·e: a threshold, a zone, a category.
// Random quantities are random points: sampleShare and sampleLevel draw a point whose logit is
// logistic around its mean (the logistic is the distribution of the space's own coordinate).
//
// The versor actions are written in closed form on the grade-1 coefficients (the sandwich V X Ṽ
// restricted to vectors); tests/test_space.cpp checks every one against the geometric product of
// syn::Multivector.

#include <cmath>
#include <cstdint>
#include <string>

namespace syn::space {

/** A vector of Cl(4, 1): x = (N, A, B, C) components, e the coefficient of e (e² = −1). */
template <class T>
struct V5 {
  T x[4];
  T e;
};

enum Channel : int { kN = 0, kA = 1, kB = 2, kC = 3 };

template <class T>
inline V5<T> v5(T n, T a, T b, T c, T e) {
  return V5<T>{{n, a, b, c}, e};
}

/** Inner product: N² = A² = B² = C² = +1, e² = −1. */
template <class T>
inline T dot5(const V5<T>& p, const V5<T>& q) {
  return p.x[0] * q.x[0] + p.x[1] * q.x[1] + p.x[2] * q.x[2] + p.x[3] * q.x[3] - p.e * q.e;
}

template <class T>
inline V5<T> add5(const V5<T>& p, const V5<T>& q) {
  return v5<T>(p.x[0] + q.x[0], p.x[1] + q.x[1], p.x[2] + q.x[2], p.x[3] + q.x[3], p.e + q.e);
}

template <class T>
inline V5<T> scale5(const V5<T>& p, T s) {
  return v5<T>(p.x[0] * s, p.x[1] * s, p.x[2] * s, p.x[3] * s, p.e * s);
}

/** X / λ with λ = −X·e the weight (1 for a normalised point). */
template <class T>
inline V5<T> unit5(const V5<T>& p) {
  return scale5(p, T(1) / p.e);
}

/** The learner L = N + e (chance 1) and its antipode −N + e (chance 0). */
template <class T>
inline V5<T> pole() {
  return v5<T>(1, 0, 0, 0, 1);
}
template <class T>
inline V5<T> antipode() {
  return v5<T>(-1, 0, 0, 0, 1);
}

/** The point of chance ½ in a channel: K + e. */
template <class T>
inline V5<T> equator(int channel) {
  return v5<T>(0, channel == kA ? T(1) : T(0), channel == kB ? T(1) : T(0), channel == kC ? T(1) : T(0), 1);
}

/** The point of chance p in a channel: (2p − 1)·N + 2√(p(1 − p))·K + e. */
template <class T>
inline V5<T> atChance(T p, int channel) {
  // Kept off the pole and the antipode, where every channel meets, so a point keeps its channel.
  const T q = p < T(1e-7) ? T(1e-7) : (p > T(1) - T(1e-7) ? T(1) - T(1e-7) : p);
  const T s = T(2) * std::sqrt(q * (T(1) - q));
  return v5<T>(T(2) * q - T(1), channel == kA ? s : T(0), channel == kB ? s : T(0), channel == kC ? s : T(0), 1);
}

/** chance(X) = 1 + ½·X̂·L = ½(1 + X_N / X_e). */
template <class T>
inline T chance(const V5<T>& p) {
  return T(0.5) * (T(1) + p.x[0] / p.e);
}

/** logit(X) = ln(chance / (1 − chance)) = ln((X_e + X_N) / (X_e − X_N)). */
template <class T>
inline T logit(const V5<T>& p) {
  return std::log((p.e + p.x[0]) / (p.e - p.x[0]));
}

/**
 * The dilation versor D = exp(−d/4 · N e) about the pole, applied to X: a boost in the (N, e)
 * plane with rapidity d/2. On a point it adds d to the logit; the result is normalised.
 */
template <class T>
inline V5<T> shift(const V5<T>& p, T d) {
  const T ch = std::cosh(T(0.5) * d), sh = std::sinh(T(0.5) * d);
  return unit5(v5<T>(ch * p.x[0] + sh * p.e, p.x[1], p.x[2], p.x[3], sh * p.x[0] + ch * p.e));
}

/** The point of logit x in a channel: the dilation of the channel's equator point. */
template <class T>
inline V5<T> atLogit(T x, int channel) {
  return shift(equator<T>(channel), x);
}

/** The channel a point lies in (its largest content component; the cognitive one if none). */
template <class T>
inline int channelOf(const V5<T>& p) {
  const T a = std::fabs(p.x[1]), b = std::fabs(p.x[2]), c = std::fabs(p.x[3]);
  return a >= b && a >= c ? kA : (b >= c ? kB : kC);
}

/** Back onto the circle (N, K): the null point of the same chance in channel K. */
template <class T>
inline V5<T> settle(const V5<T>& p, int channel) {
  return atChance(chance(p), channel);
}

/** The mixture (1 − t)·X̂ + t·Ŷ of two normalised vectors; its chance is the same mixture. */
template <class T>
inline V5<T> mixture(const V5<T>& p, const V5<T>& q, T t) {
  return add5(scale5(unit5(p), T(1) - t), scale5(unit5(q), t));
}

/** Mixture towards the learner, settled: chance ↦ chance + t·(1 − chance). */
template <class T>
inline V5<T> raise(const V5<T>& p, T t) {
  return settle(mixture(p, pole<T>(), t), channelOf(p));
}

/** Mixture towards the antipode, settled: chance ↦ chance·(1 − t). */
template <class T>
inline V5<T> lower(const V5<T>& p, T t) {
  return settle(mixture(p, antipode<T>(), t), channelOf(p));
}

/** Additive change of the reading by d (clamped to [0, 1]) as a raise or a lower. */
template <class T>
inline V5<T> nudge(const V5<T>& p, T d) {
  const T c = chance(p);
  if (d >= T(0)) return c >= T(1) ? p : raise(p, d >= T(1) - c ? T(1) : d / (T(1) - c));
  return c <= T(0) ? p : lower(p, -d >= c ? T(1) : -d / c);
}

/** The mixture of X and Y with weight t on Y, settled in X's channel. */
template <class T>
inline V5<T> blend(const V5<T>& p, const V5<T>& q, T t) {
  return settle(mixture(p, q, t), channelOf(p));
}

/** Rotor in the coordinate plane (i, j) of the Euclidean directions: i turns towards j by α. */
template <class T>
inline V5<T> rotate(const V5<T>& p, int i, int j, T alpha) {
  V5<T> r = p;
  const T c = std::cos(alpha), s = std::sin(alpha);
  r.x[i] = c * p.x[i] - s * p.x[j];
  r.x[j] = s * p.x[i] + c * p.x[j];
  return r;
}

/**
 * The rotor exp(−tθ/2 · B̂), B̂ the unit bivector of the plane of x̂ and ŷ, applied to the
 * normalised point X: it turns X by the fraction t of the angle θ between the two points.
 */
template <class T>
inline V5<T> towards(const V5<T>& p, const V5<T>& q, T t) {
  const V5<T> a = unit5(p), b = unit5(q);
  T c = a.x[0] * b.x[0] + a.x[1] * b.x[1] + a.x[2] * b.x[2] + a.x[3] * b.x[3];
  c = c > T(1) ? T(1) : (c < T(-1) ? T(-1) : c);
  const T theta = std::acos(c), s = std::sin(theta);
  if (s < T(1e-6)) return a;
  const T k0 = std::cos(t * theta) - c * std::sin(t * theta) / s, k1 = std::sin(t * theta) / s;
  return v5<T>(k0 * a.x[0] + k1 * b.x[0], k0 * a.x[1] + k1 * b.x[1], k0 * a.x[2] + k1 * b.x[2], k0 * a.x[3] + k1 * b.x[3], 1);
}

/** 1 + ½·X̂·Ŷ: cos²(θ/2) for two points, θ the angle between them. */
template <class T>
inline T similarity(const V5<T>& p, const V5<T>& q) {
  return T(1) + T(0.5) * dot5(unit5(p), unit5(q));
}

/** The cap of the points within angle ρ of a point: S = ĉ + cos ρ·e. */
template <class T>
inline V5<T> cap(const V5<T>& centre, T rho) {
  const V5<T> c = unit5(centre);
  return v5<T>(c.x[0], c.x[1], c.x[2], c.x[3], std::cos(rho));
}

/** The cap of the points of chance at least p: L − 2(1 − p)·e. */
template <class T>
inline V5<T> chanceCap(T p) {
  return v5<T>(1, 0, 0, 0, T(2) * p - T(1));
}

/** X̂·S: positive inside the cap, zero on its boundary, negative outside. */
template <class T>
inline T side(const V5<T>& p, const V5<T>& s) {
  return dot5(unit5(p), s);
}
template <class T>
inline bool inside(const V5<T>& p, const V5<T>& s) {
  return side(p, s) >= T(0);
}

/**
 * Inverse stereographic projection of a point u of R³ (scaled by 1/scale) onto S³, from the
 * antipode: u = 0 is the pole, and small neighbourhoods keep their shape (it is conformal).
 */
template <class T>
inline V5<T> lift(T u1, T u2, T u3, T scale) {
  const T a = u1 / scale, b = u2 / scale, c = u3 / scale;
  const T r2 = a * a + b * b + c * c;
  const T k = T(1) / (T(1) + r2);
  return v5<T>((T(1) - r2) * k, T(2) * a * k, T(2) * b * k, T(2) * c * k, 1);
}

/** The point in the direction of the Euclidean part of a vector (e.g. a sum of points: their centre). */
template <class T>
inline V5<T> pointOf(const V5<T>& p) {
  const T n = std::sqrt(p.x[0] * p.x[0] + p.x[1] * p.x[1] + p.x[2] * p.x[2] + p.x[3] * p.x[3]);
  return v5<T>(p.x[0] / n, p.x[1] / n, p.x[2] / n, p.x[3] / n, 1);
}

// ---------------------------------------------------------------- random points

/** ln(u / (1 − u)): the logistic variate of a uniform u. */
template <class T>
inline T logisticOf(T u) {
  return std::log(u / (T(1) - u));
}

/** sd of the standard logistic is π/√3; this scales a logistic variate to sd 1. */
constexpr float kLogisticUnit = 0.5513288954217921f;

/** A random level: the point of logit mean + sd·(logistic of u, scaled to sd 1). */
template <class T>
inline V5<T> sampleLevel(T u, T mean, T sd, int channel) {
  return atLogit(mean + sd * T(kLogisticUnit) * logisticOf(u), channel);
}

/** A random share in (0, 1) around m with spread sd (on the chance scale, near the mean). */
template <class T>
inline V5<T> sampleShare(T u, T m, T sd, int channel) {
  const T mm = m < T(0.02) ? T(0.02) : (m > T(0.98) ? T(0.98) : m);
  const T spread = sd / (mm * (T(1) - mm));
  return atLogit(std::log(mm / (T(1) - mm)) + spread * T(kLogisticUnit) * logisticOf(u), channel);
}

/** The WGSL twin of this header (every function above, same names, f32). */
const std::string& spaceWgsl();

}  // namespace syn::space
