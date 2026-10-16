#include "syn/classroom/learner.hpp"

#include <algorithm>
#include <stdexcept>

#include "syn/core/math.hpp"
#include "syn/core/random.hpp"

namespace syn {

PopulationSpec PopulationSpec::children() {
  PopulationSpec p;
  p.name = "primary school (9)";
  p.age = 9.0;
  p.anxietyMean = 0.3;
  p.growthMean = 0.6;
  p.efficacyMean = 0.6;
  p.intrinsicMean = 0.4;
  return p;
}

PopulationSpec PopulationSpec::secondary() { return PopulationSpec{}; }

PopulationSpec PopulationSpec::adults() {
  PopulationSpec p;
  p.name = "adult learners (35)";
  p.age = 35.0;
  p.ageSd = 8.0;
  p.anxietyMean = 0.45;
  p.growthMean = 0.45;
  p.efficacyMean = 0.5;
  p.intrinsicMean = 0.55;
  return p;
}

double adultness(double age) { return clamp01((age - 12.0) / 10.0); }

Learner sampleLearner(const PopulationSpec& s, Rng& rng) {
  Learner l;
  l.age = std::max(4.0, rng.normal(s.age, s.ageSd));
  l.capital = clamp01(rng.normal(s.capitalMean, s.capitalSd));
  l.ability = rng.normal(s.priorMean, s.priorSd) + s.capitalPrior * (l.capital - s.capitalMean);
  l.anxiety = clamp01(rng.normal(s.anxietyMean, s.anxietySd));
  l.growth = clamp01(rng.normal(s.growthMean, s.growthSd));
  l.efficacy = clamp01(rng.normal(s.efficacyMean, s.efficacySd));
  l.intrinsic = clamp01(rng.normal(s.intrinsicMean, s.intrinsicSd));
  l.physiological = clamp01(rng.normal(0.85, 0.1));
  return l;
}

std::vector<Learner> samplePopulation(const PopulationSpec& spec, int n, std::uint64_t seed) {
  if (n < 0) throw std::invalid_argument("samplePopulation: negative size");
  std::vector<Learner> out;
  for (int i = 0; i < n; ++i) {
    Rng rng = Rng::stream(seed, static_cast<std::uint64_t>(i));
    out.push_back(sampleLearner(spec, rng));
  }
  return out;
}

}  // namespace syn
