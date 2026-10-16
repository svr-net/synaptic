#include "syn/cognitivism/interleaving.hpp"

#include <cmath>
#include <stdexcept>

#include "syn/core/random.hpp"

namespace syn {

namespace {

PracticeOutcome practise(const InterleavingParams& p, bool interleaved, std::uint64_t seed) {
  const int k = p.categories, n = p.categories * p.problemsPerCategory;
  PracticeOutcome out;
  out.practiceAccuracy.assign(n, 0.0);
  const double chance = 1.0 / k;
  auto choice = [&](double d) { return chance + (1.0 - chance) * (1.0 - std::exp(-d)); };
  for (int i = 0; i < p.learners; ++i) {
    // The same learner (same learning rates) under both conditions: common random numbers.
    Rng rng = Rng::stream(seed, static_cast<std::uint64_t>(i));
    const double eta = p.executionRate * std::exp(p.heterogeneity * rng.normal());
    const double delta = p.discriminationRate * std::exp(p.heterogeneity * rng.normal());
    std::vector<double> e(k, p.initialExecution);
    double d = 0.0;
    for (int t = 0; t < n; ++t) {
      const int type = interleaved ? t % k : t / p.problemsPerCategory;
      const bool mustChoose = interleaved || t % p.problemsPerCategory == 0;
      const double acc = (mustChoose ? choice(d) : 1.0) * e[type];
      out.practiceAccuracy[t] += acc / p.learners;
      if (mustChoose) d += delta;
      e[type] += std::min(1.0, eta) * (1.0 - e[type]);
    }
    double meanE = 0.0;
    for (double x : e) meanE += x / k;
    out.choiceAccuracy += choice(d) / p.learners;
    out.execution += meanE / p.learners;
    out.testAccuracy += choice(d) * meanE / p.learners;
  }
  for (double a : out.practiceAccuracy) out.practiceMean += a / n;
  return out;
}

}  // namespace

InterleavingResult simulateInterleaving(const InterleavingParams& p, std::uint64_t seed) {
  if (p.categories < 2 || p.problemsPerCategory < 1 || p.learners < 1) throw std::invalid_argument("simulateInterleaving: bad sizes");
  return {practise(p, false, seed), practise(p, true, seed)};
}

}  // namespace syn
