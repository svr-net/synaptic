#include "syn/constructivism/equilibration.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <stdexcept>

#include "syn/core/random.hpp"

namespace syn {

const char* adaptationName(Adaptation a) {
  switch (a) {
    case Adaptation::Assimilation: return "assimilation";
    case Adaptation::Disequilibrium: return "disequilibrium";
    case Adaptation::Accommodation: return "accommodation";
    case Adaptation::Avoidance: return "avoidance";
  }
  return "?";
}

SchemaLearner::SchemaLearner(EquilibrationParams params, int dimensions)
    : p_(params), g_(dimensions), capShift_(1.0 - std::cos(params.assimilationRadius)) {
  if (!(p_.assimilationRadius > 0.0 && p_.assimilationRadius < M_PI) || p_.persistence < 0.0 || p_.persistence >= 1.0)
    throw std::invalid_argument("EquilibrationParams: need radius in (0, π) and persistence in [0, 1)");
}

Multivector SchemaLearner::lift(const std::vector<double>& features) const { return g_.liftStereographic(features, p_.featureScale); }

void SchemaLearner::addSchema(const std::vector<double>& prototype, int label) { schemas_.push_back({lift(prototype), label, 0}); }

std::vector<double> SchemaLearner::prototypeFeatures(std::size_t schema) const {
  return g_.projectStereographic(schemas_.at(schema).prototype, p_.featureScale);
}

int SchemaLearner::nearest(const Multivector& x) const {
  // X·P = cos θ − 1 grows as the geodesic angle shrinks.
  int best = -1;
  double bestDot = -std::numeric_limits<double>::infinity();
  for (std::size_t i = 0; i < schemas_.size(); ++i) {
    const double d = g_.dot(x, schemas_[i].prototype);
    if (d > bestDot) {
      bestDot = d;
      best = static_cast<int>(i);
    }
  }
  return best;
}

bool SchemaLearner::fits(const Multivector& x, const Schema& s) const {
  // Inside the cap P − (1 − cos ρ)·e: X·S = X·P + (1 − cos ρ) ≥ 0 (X·e = −1 for a normalised point).
  return g_.inside(x, s.prototype - g_.e() * capShift_);
}

int SchemaLearner::classify(const Multivector& x) const {
  const int i = nearest(x);
  return i < 0 ? -1 : schemas_[i].label;
}

int SchemaLearner::classify(const std::vector<double>& x) const { return classify(lift(x)); }

Adaptation SchemaLearner::observe(const Observation& obs, double u) {
  const Multivector x = lift(obs.features);
  int target = nearest(x);
  bool ok = target >= 0 && fits(x, schemas_[target]) && (obs.label < 0 || schemas_[target].label == obs.label);
  if (!ok && obs.label >= 0) {
    // A schema with the right label whose cap holds the experience: the structure already fits.
    double best = -std::numeric_limits<double>::infinity();
    for (std::size_t i = 0; i < schemas_.size(); ++i) {
      const double d = g_.dot(x, schemas_[i].prototype);
      if (schemas_[i].label == obs.label && fits(x, schemas_[i]) && d > best) {
        best = d;
        target = static_cast<int>(i);
        ok = true;
      }
    }
  }
  if (ok) {
    Schema& s = schemas_[target];
    s.prototype = g_.move(g_.rotorTowards(s.prototype, x, p_.learningRate), s.prototype);
    ++s.assimilated;
    d_ *= p_.persistence;
    return Adaptation::Assimilation;
  }
  if (u < p_.avoidance) {
    d_ *= p_.persistence;
    return Adaptation::Avoidance;
  }
  d_ = p_.persistence * d_ + (1.0 - p_.persistence);
  conflicts_.emplace_back(x, obs.label);
  if (d_ < p_.accommodationThreshold) return Adaptation::Disequilibrium;

  // Accommodate: a new schema centred on the conflicting experiences, named by the feedback.
  std::map<int, int> votes;
  for (const auto& c : conflicts_)
    if (c.second >= 0) ++votes[c.second];
  // Without feedback the new category has no name yet: give it a label no real category uses.
  int label = 1000;
  for (const auto& s : schemas_) label = std::max(label, s.label + 1);
  int bestVotes = 0;
  for (const auto& [l, v] : votes)
    if (v > bestVotes) {
      bestVotes = v;
      label = l;
    }
  std::vector<Multivector> members;
  for (const auto& c : conflicts_)
    if (c.second < 0 || c.second == label) members.push_back(c.first);
  schemas_.push_back({g_.centre(members), label, static_cast<int>(members.size())});
  conflicts_.clear();
  d_ *= 0.3;
  return Adaptation::Accommodation;
}

EquilibrationRun simulateEquilibration(const EquilibrationParams& params, const AnimalWorld& w, std::uint64_t seed) {
  if (w.observations < 1 || w.learners < 1) throw std::invalid_argument("simulateEquilibration: bad sizes");
  const std::vector<double> dog = {1.0, 0.0, 0.0}, cat = {0.3, 1.0, 1.0};
  enum { kDog = 0, kCat = 1 };
  auto draw = [&](Rng& rng, int kind) {
    const auto& c = kind == kDog ? dog : cat;
    std::vector<double> x(c.size());
    for (std::size_t k = 0; k < c.size(); ++k) x[k] = c[k] + w.noise * rng.normal();
    return x;
  };
  // A fixed test set of dogs and cats.
  Rng testRng(seed ^ 0x7e57ULL);
  std::vector<Observation> test;
  for (int i = 0; i < 100; ++i) test.push_back({draw(testRng, i % 2), i % 2});

  std::vector<Multivector> lifted;  // the test set on the sphere
  EquilibrationRun run;
  const auto n = static_cast<std::size_t>(w.observations);
  run.disequilibrium.assign(n, 0.0);
  run.accuracy.assign(n, 0.0);
  run.schemaCount.assign(n, 0.0);
  run.labelledShare.assign(n, 0.0);
  for (int l = 0; l < w.learners; ++l) {
    Rng rng = Rng::stream(seed, static_cast<std::uint64_t>(l));
    SchemaLearner learner(params);
    learner.addSchema(dog, kDog);
    if (lifted.empty())
      for (const auto& o : test) lifted.push_back(learner.lift(o.features));
    for (std::size_t t = 0; t < n; ++t) {
      const double uKind = rng.uniform(), uFeedback = rng.uniform(), uAvoid = rng.uniform();
      const int kind = static_cast<int>(t) >= w.catsFrom && uKind < w.catShare ? kCat : kDog;
      Observation obs{draw(rng, kind), uFeedback < w.feedback ? kind : -1};
      switch (learner.observe(obs, uAvoid)) {
        case Adaptation::Assimilation: run.assimilations += 1; break;
        case Adaptation::Disequilibrium: run.conflicts += 1; break;
        case Adaptation::Accommodation: run.conflicts += 1; run.accommodations += 1; break;
        case Adaptation::Avoidance: run.avoidances += 1; break;
      }
      int correct = 0;
      for (std::size_t k = 0; k < test.size(); ++k) correct += learner.classify(lifted[k]) == test[k].label ? 1 : 0;
      run.accuracy[t] += correct / static_cast<double>(test.size()) / w.learners;
      run.disequilibrium[t] += learner.disequilibrium() / w.learners;
      run.schemaCount[t] += static_cast<double>(learner.schemas().size()) / w.learners;
      run.labelledShare[t] += (obs.label >= 0 ? 1.0 : 0.0) / w.learners;
    }
  }
  run.assimilations /= w.learners;
  run.accommodations /= w.learners;
  run.avoidances /= w.learners;
  run.conflicts /= w.learners;
  run.finalAccuracy = run.accuracy.back();
  return run;
}

}  // namespace syn
