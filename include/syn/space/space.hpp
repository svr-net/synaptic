#pragma once

// The space: one pure elliptic conformal geometric algebra in which every model of the library
// lives. It exists once, in WGSL (spaceWgsl()): every kernel includes it, on WebGPU and on the
// software GPU alike.
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
// restricted to vectors); tests/test_space.cpp runs every one on the software GPU and checks it
// against the geometric product of syn::Multivector.

#include <string>

namespace syn::space {

/** The space as WGSL: struct V5, the channels kN kA kB kC, and every function named above. */
const std::string& spaceWgsl();

}  // namespace syn::space
