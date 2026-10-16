#pragma once

// Pure elliptic conformal geometric algebra: the conformal model of the n-sphere Sⁿ.
//
// The algebra is Cl(n+1, 1): the n+1 Euclidean directions of the space Sⁿ sits in, plus one
// extra vector e with e² = −1. There is no origin and no point at infinity (no e₀/e∞ pair):
// every point is a point of the sphere, and every distance is a geodesic angle.
//   point      X = λ(x + e),  x ∈ Sⁿ           X² = 0 (null vector), normalised when λ = 1
//   distance   X·Y = cos θ − 1 = −2 sin²(θ/2)   θ the geodesic angle between the points
//   cap        S = c + cos ρ · e               the points within angle ρ of c; S² = sin²ρ
//   inside     X·S = cos θ − cos ρ             > 0 inside, 0 on the boundary, < 0 outside
//   rotation   R = exp(−θ/2 · B̂), B̂ a unit Euclidean bivector: an isometry of Sⁿ
//   dilation   D = exp(β/2 · n e): the conformal map fixing ±n that scales tan(φ/2) by e^β,
//              φ the angle from n
// Versors act by the sandwich V X Ṽ. Rotations keep normalised points normalised; a dilation
// rescales λ, and `normalise` restores it.

#include <vector>

#include "syn/core/multivector.hpp"

namespace syn {

class EllipticCGA {
 public:
  /** The model of Sⁿ, n ≥ 1 (Cl(n+1, 1), so n + 2 ≤ 10). */
  explicit EllipticCGA(int n);

  int n() const { return n_; }
  /** Index of e (the vector with e² = −1). */
  int eIndex() const { return n_ + 1; }
  Multivector e() const;

  /** Normalised point x + e for a direction x ∈ R^{n+1} (normalised to the unit sphere). */
  Multivector point(const std::vector<double>& direction) const;
  /** Point at geodesic angle φ from `from` towards `towards` (both directions, any length). */
  Multivector pointAt(const std::vector<double>& from, const std::vector<double>& towards, double phi) const;
  /** λ = −X·e: the weight of a point (1 when normalised). */
  double weight(const Multivector& x) const;
  /** X / λ. */
  Multivector normalise(const Multivector& x) const;
  /** The unit direction x of a point. */
  std::vector<double> direction(const Multivector& x) const;
  /** True when X is (numerically) a null vector of this algebra. */
  bool isPoint(const Multivector& x, double tol = 1e-9) const;

  /** Inner product of two vectors (their grade-1 parts). */
  double dot(const Multivector& a, const Multivector& b) const;
  /** Geodesic angle between two points: atan2(|x∧y|, 1 + X·Y) on normalised points (accurate near 0 and π). */
  double angle(const Multivector& x, const Multivector& y) const;

  /** Cap of the points within angle ρ (0 < ρ < π) of a centre point. */
  Multivector cap(const Multivector& centre, double rho) const;
  /** X·S for a normalised point and a normalised cap (> 0 inside). */
  double capSide(const Multivector& x, const Multivector& s) const;
  bool inside(const Multivector& x, const Multivector& s) const { return capSide(x, s) >= 0.0; }

  /** Rotor moving point a towards b along their geodesic by the fraction t (t = 1 reaches b). */
  Multivector rotorTowards(const Multivector& a, const Multivector& b, double t) const;
  /** Dilation versor exp(β/2 · n e) fixing ±n: tan(φ/2) ↦ e^β tan(φ/2). */
  Multivector dilation(const std::vector<double>& pole, double beta) const;
  /** V X Ṽ, normalised to a point. */
  Multivector move(const Multivector& versor, const Multivector& x) const;

  /**
   * Centre of a set of points: the direction of their sum. The sum of null vectors is a
   * vector whose Euclidean part points to the extrinsic mean of the points on the sphere.
   */
  Multivector centre(const std::vector<Multivector>& points) const;

  /** Inverse stereographic projection of u ∈ Rⁿ (scaled by 1/scale) onto Sⁿ, as a point. */
  Multivector liftStereographic(const std::vector<double>& u, double scale = 1.0) const;
  /** Stereographic projection of a point back to Rⁿ (from the pole −e_{n+1}). */
  std::vector<double> projectStereographic(const Multivector& x, double scale = 1.0) const;

 private:
  std::vector<double> unit(const std::vector<double>& v) const;
  int n_;
};

}  // namespace syn
