#pragma once

// Piaget's equilibration: assimilation, disequilibrium and accommodation.
//
// A learner's schemas are prototypes in a feature space, each with a label ("dog"). An
// observation that falls near a schema and does not contradict it is assimilated: the schema
// absorbs it and drifts slightly towards it, but the learner's understanding does not change.
// An observation that contradicts the nearest schema (the world says "that is a cat") or lies
// far from every schema creates cognitive conflict. Conflict accumulates as disequilibrium
// D ← ρ·D + (1 − ρ)·1. When D passes a threshold the learner accommodates: a new schema is
// built from the conflicting observations, and equilibrium is restored. Only accommodation
// changes the structure of knowledge, so it is where learning happens.
//
// Disequilibrium is uncomfortable. A learner who avoids discomfort ignores a conflicting
// observation with probability `avoidance`; a supportive, low-stakes environment lowers it.
//
// Geometry (pure elliptic CGA, see core/elliptic_cga.hpp). Feature vectors are lifted onto the
// sphere S³ by inverse stereographic projection (conformal: it keeps the shape of small
// neighbourhoods), and from then on everything is a CGA operation:
//   an experience   a null vector X (a point of S³)
//   a schema        a prototype point P and its assimilation cap S = p + cos ρ·e
//   fits?           X·S ≥ 0 (inside the cap); nearest schema = largest X·P
//   assimilation    the prototype is rotated towards X by the rotor R = exp(−η θ/2 · B̂),
//                   P ← R P R̃, a fraction η of the geodesic between them
//   accommodation   a new prototype at the centre of the conflicting experiences: the direction
//                   of the sum of their null vectors

#include <cstdint>
#include <string>
#include <vector>

#include "syn/core/elliptic_cga.hpp"
#include "syn/core/multivector.hpp"

namespace syn {

struct Observation {
  std::vector<double> features;
  int label = -1;  // -1: no feedback from the world, otherwise the true category
};

struct Schema {
  Multivector prototype;  // a normalised point of S³
  int label = 0;
  int assimilated = 0;
};

enum class Adaptation { Assimilation, Disequilibrium, Accommodation, Avoidance };

const char* adaptationName(Adaptation a);

struct EquilibrationParams {
  double assimilationRadius = 1.25; // angular radius ρ (radians) of each schema's assimilation cap
  double featureScale = 2.0;        // scale of the stereographic lift of feature space onto S³
  double learningRate = 0.08;       // prototype drift on assimilation
  double persistence = 0.7;         // ρ
  double accommodationThreshold = 0.45;
  double avoidance = 0.0;           // P(ignore a conflicting observation)
};

class SchemaLearner {
 public:
  /** dimensions: number of features (the learner works on the sphere S^dimensions). */
  explicit SchemaLearner(EquilibrationParams params, int dimensions = 3);

  /** Starts with an initial schema (e.g. the child already knows dogs), given in feature space. */
  void addSchema(const std::vector<double>& prototype, int label);

  /** Label the learner would give (nearest schema), or -1 when it has none. */
  int classify(const std::vector<double>& x) const;
  /** The same for an experience already lifted onto the sphere. */
  int classify(const Multivector& x) const;
  /** A schema's prototype projected back to feature space. */
  std::vector<double> prototypeFeatures(std::size_t schema) const;
  /** The lift of a feature vector onto S³. */
  Multivector lift(const std::vector<double>& features) const;
  const EllipticCGA& geometry() const { return g_; }

  /** Processes one observation; `u` is a uniform draw used for avoidance. */
  Adaptation observe(const Observation& obs, double u);

  double disequilibrium() const { return d_; }
  const std::vector<Schema>& schemas() const { return schemas_; }

 private:
  int nearest(const Multivector& x) const;
  bool fits(const Multivector& x, const Schema& s) const;
  EquilibrationParams p_;
  EllipticCGA g_;
  double capShift_;  // 1 − cos ρ: the cap is P − (1 − cos ρ)·e
  std::vector<Schema> schemas_;
  std::vector<std::pair<Multivector, int>> conflicts_;  // lifted experience and its label
  double d_ = 0.0;
};

struct EquilibrationRun {
  std::vector<double> disequilibrium;  // mean over learners, after each observation
  std::vector<double> accuracy;        // mean test accuracy after each observation
  std::vector<double> schemaCount;     // mean number of schemas
  std::vector<double> labelledShare;   // share of observations that carried feedback
  double assimilations = 0.0, accommodations = 0.0, avoidances = 0.0, conflicts = 0.0;  // per learner
  double finalAccuracy = 0.0;
};

/**
 * The book's example: a child who knows dogs meets dogs and cats. Features are
 * [size, ear pointiness, sound (bark → meow)]. `feedback` is the probability that an adult
 * names the animal; cats appear from observation `catsFrom` onwards with share `catShare`.
 */
struct AnimalWorld {
  int observations = 120;
  int catsFrom = 20;
  double catShare = 0.5;
  double feedback = 0.3;
  double noise = 0.35;
  int learners = 200;
};

EquilibrationRun simulateEquilibration(const EquilibrationParams& params, const AnimalWorld& world, std::uint64_t seed);

}  // namespace syn
