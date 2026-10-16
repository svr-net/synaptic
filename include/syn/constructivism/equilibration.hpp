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

#include <cstdint>
#include <string>
#include <vector>

namespace syn {

struct Observation {
  std::vector<double> features;
  int label = -1;  // -1: no feedback from the world, otherwise the true category
};

struct Schema {
  std::vector<double> prototype;
  int label = 0;
  int assimilated = 0;
};

enum class Adaptation { Assimilation, Disequilibrium, Accommodation, Avoidance };

const char* adaptationName(Adaptation a);

struct EquilibrationParams {
  double assimilationRadius = 1.6;  // distance within which an observation can be assimilated
  double learningRate = 0.08;       // prototype drift on assimilation
  double persistence = 0.7;         // ρ
  double accommodationThreshold = 0.45;
  double avoidance = 0.0;           // P(ignore a conflicting observation)
};

class SchemaLearner {
 public:
  explicit SchemaLearner(EquilibrationParams params);

  /** Starts with an initial schema (e.g. the child already knows dogs). */
  void addSchema(std::vector<double> prototype, int label);

  /** Label the learner would give (nearest schema), or -1 when it has none. */
  int classify(const std::vector<double>& x) const;

  /** Processes one observation; `u` is a uniform draw used for avoidance. */
  Adaptation observe(const Observation& obs, double u);

  double disequilibrium() const { return d_; }
  const std::vector<Schema>& schemas() const { return schemas_; }

 private:
  int nearest(const std::vector<double>& x, double* distance) const;
  EquilibrationParams p_;
  std::vector<Schema> schemas_;
  std::vector<Observation> conflicts_;
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
