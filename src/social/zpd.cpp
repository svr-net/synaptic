#include "syn/social/zpd.hpp"

#include <cmath>
#include <stdexcept>

#include "syn/core/math.hpp"
#include "syn/core/random.hpp"

namespace syn {

const char* zoneName(Zone z) {
  switch (z) {
    case Zone::Independent: return "can do alone";
    case Zone::Proximal: return "zone of proximal development";
    case Zone::Beyond: return "cannot do yet";
  }
  return "?";
}

const char* taskPolicyName(TaskPolicy p) {
  switch (p) {
    case TaskPolicy::TooEasy: return "too easy (review)";
    case TaskPolicy::ClassAverage: return "one level for the whole class";
    case TaskPolicy::TooHard: return "too hard";
    case TaskPolicy::Zpd: return "targeted at each learner's ZPD";
  }
  return "?";
}

double successAlone(double theta, double b) { return logistic(theta - b); }
double successAssisted(double theta, double b, double scaffold) { return logistic(theta + scaffold - b); }

Zone zoneOf(double theta, double b, const ZpdParams& p) {
  if (successAlone(theta, b) >= p.independentThreshold) return Zone::Independent;
  if (successAssisted(theta, b, p.scaffold) >= p.assistedThreshold) return Zone::Proximal;
  return Zone::Beyond;
}

double expectedGain(double theta, double b, const ZpdParams& p) {
  return p.learningRate * successAssisted(theta, b, p.scaffold) * (1.0 - successAlone(theta, b));
}

std::pair<double, double> zpdBand(double theta, const ZpdParams& p) {
  // Independent iff b ≤ θ − logit(p_hi); reachable with help iff b ≤ θ + s − logit(p_lo).
  return {theta - logit(p.independentThreshold), theta + p.scaffold - logit(p.assistedThreshold)};
}

TutoringRun simulateTutoring(const ZpdParams& zpd, const TutoringParams& tp, TaskPolicy policy, std::uint64_t seed) {
  if (tp.learners < 1 || tp.sessions < 1) throw std::invalid_argument("simulateTutoring: bad sizes");
  TutoringRun run;
  run.policy = taskPolicyName(policy);
  const auto n = static_cast<std::size_t>(tp.sessions);
  run.zoneShare.assign(3, std::vector<double>(n, 0.0));
  std::vector<std::vector<double>> ability(n, std::vector<double>(tp.learners));
  const auto band = zpdBand(0.0, zpd);
  const double zpdCentre = 0.5 * (band.first + band.second);  // offset of the ZPD's middle from θ
  for (int i = 0; i < tp.learners; ++i) {
    Rng rng = Rng::stream(seed, static_cast<std::uint64_t>(i));
    double theta = rng.normal(tp.abilityMean, tp.abilitySd);
    const double start = theta;
    for (std::size_t s = 0; s < n; ++s) {
      const double estimate = theta + tp.assessmentNoise * rng.normal();
      const double classLevel = tp.classDifficulty + tp.curriculumPace * static_cast<double>(s);
      double b = classLevel;
      switch (policy) {
        case TaskPolicy::TooEasy: b = theta - 2.5; break;
        case TaskPolicy::ClassAverage: b = classLevel; break;
        case TaskPolicy::TooHard: b = theta + 5.0; break;
        case TaskPolicy::Zpd: b = estimate + zpdCentre; break;
      }
      ZpdParams here = zpd;
      if (tp.fadeScaffold) here.scaffold = zpd.scaffold * (0.4 + 0.6 * (1.0 - successAlone(theta, b)));
      ability[s][i] = theta;
      run.zoneShare[static_cast<int>(zoneOf(theta, b, zpd))][s] += 1.0 / tp.learners;
      theta += expectedGain(theta, b, here) * (1.0 + 0.2 * rng.normal());
    }
    run.finalGain += (theta - start) / tp.learners;
  }
  for (const auto& a : ability) {
    run.meanAbility.push_back(mean(a));
    run.q10Ability.push_back(quantile(a, 0.1));
    run.q90Ability.push_back(quantile(a, 0.9));
  }
  return run;
}

}  // namespace syn
