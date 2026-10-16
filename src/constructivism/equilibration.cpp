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

namespace {
double distance(const std::vector<double>& a, const std::vector<double>& b) {
  if (a.size() != b.size()) throw std::invalid_argument("equilibration: feature dimension mismatch");
  double s = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) s += (a[i] - b[i]) * (a[i] - b[i]);
  return std::sqrt(s);
}
}  // namespace

SchemaLearner::SchemaLearner(EquilibrationParams params) : p_(params) {
  if (p_.assimilationRadius <= 0.0 || p_.persistence < 0.0 || p_.persistence >= 1.0)
    throw std::invalid_argument("EquilibrationParams: need radius > 0 and persistence in [0, 1)");
}

void SchemaLearner::addSchema(std::vector<double> prototype, int label) { schemas_.push_back({std::move(prototype), label, 0}); }

int SchemaLearner::nearest(const std::vector<double>& x, double* dist) const {
  int best = -1;
  double bestD = std::numeric_limits<double>::infinity();
  for (std::size_t i = 0; i < schemas_.size(); ++i) {
    const double d = distance(x, schemas_[i].prototype);
    if (d < bestD) {
      bestD = d;
      best = static_cast<int>(i);
    }
  }
  if (dist) *dist = bestD;
  return best;
}

int SchemaLearner::classify(const std::vector<double>& x) const {
  const int i = nearest(x, nullptr);
  return i < 0 ? -1 : schemas_[i].label;
}

Adaptation SchemaLearner::observe(const Observation& obs, double u) {
  double dist = 0.0;
  int target = nearest(obs.features, &dist);
  bool fits = target >= 0 && dist <= p_.assimilationRadius && (obs.label < 0 || schemas_[target].label == obs.label);
  if (!fits && obs.label >= 0) {
    // A schema with the right label nearby: the knowledge structure already fits.
    double best = p_.assimilationRadius;
    for (std::size_t i = 0; i < schemas_.size(); ++i) {
      const double d = distance(obs.features, schemas_[i].prototype);
      if (schemas_[i].label == obs.label && d <= best) {
        best = d;
        target = static_cast<int>(i);
        fits = true;
      }
    }
  }
  if (fits) {
    Schema& s = schemas_[target];
    for (std::size_t k = 0; k < s.prototype.size(); ++k) s.prototype[k] += p_.learningRate * (obs.features[k] - s.prototype[k]);
    ++s.assimilated;
    d_ *= p_.persistence;
    return Adaptation::Assimilation;
  }
  if (u < p_.avoidance) {
    d_ *= p_.persistence;
    return Adaptation::Avoidance;
  }
  d_ = p_.persistence * d_ + (1.0 - p_.persistence);
  conflicts_.push_back(obs);
  if (d_ < p_.accommodationThreshold) return Adaptation::Disequilibrium;

  // Accommodate: a new schema from the conflicting experiences, named by the feedback received.
  std::map<int, int> votes;
  // Without feedback the new category has no name yet: give it a label no real category uses.
  int nextLabel = 1000;
  for (const auto& s : schemas_) nextLabel = std::max(nextLabel, s.label + 1);
  for (const auto& c : conflicts_)
    if (c.label >= 0) ++votes[c.label];
  int label = nextLabel;
  int bestVotes = 0;
  for (const auto& [l, v] : votes)
    if (v > bestVotes) {
      bestVotes = v;
      label = l;
    }
  std::vector<double> proto(obs.features.size(), 0.0);
  int used = 0;
  for (const auto& c : conflicts_)
    if (c.label < 0 || c.label == label) {
      for (std::size_t k = 0; k < proto.size(); ++k) proto[k] += c.features[k];
      ++used;
    }
  for (double& x : proto) x /= used;
  schemas_.push_back({proto, label, used});
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
      for (const auto& o : test) correct += learner.classify(o.features) == o.label ? 1 : 0;
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
