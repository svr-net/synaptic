#include "syn/andragogy/andragogy.hpp"

#include <algorithm>
#include <numeric>
#include <stdexcept>

#include "syn/core/math.hpp"
#include "syn/core/random.hpp"

namespace syn {

LearnerOrientation LearnerOrientation::child() { return {{0.15, 0.1, 0.15, 0.2, 0.35, 0.3}}; }
LearnerOrientation LearnerOrientation::adolescent() { return {{0.45, 0.35, 0.4, 0.45, 0.45, 0.6}}; }
LearnerOrientation LearnerOrientation::adult() { return {{0.85, 0.85, 0.85, 0.85, 0.8, 0.9}}; }

AndragogicalDesign AndragogicalDesign::pedagogical() { return {{0.1, 0.1, 0.2, 0.2, 0.2, 0.3}}; }
AndragogicalDesign AndragogicalDesign::andragogical() { return {{0.9, 0.9, 0.9, 0.9, 0.85, 0.9}}; }

const char* assumptionName(int k) {
  static const char* names[kAndragogyDimensions] = {"self-directed self-concept", "reservoir of experience", "readiness tied to social roles",
                                                    "performance-centred orientation", "internal motivation", "need to know why"};
  return k >= 0 && k < kAndragogyDimensions ? names[k] : "?";
}

const char* designFeatureName(int k) {
  static const char* names[kAndragogyDimensions] = {"learner input", "use of experience", "relevance to roles",
                                                    "immediate application", "intrinsic appeal (no rewards)", "transparency of purpose"};
  return k >= 0 && k < kAndragogyDimensions ? names[k] : "?";
}

std::array<double, kAndragogyDimensions> engagementTerms(const LearnerOrientation& o, const AndragogicalDesign& d,
                                                         const AndragogyParams& p) {
  std::array<double, kAndragogyDimensions> t{};
  for (int k = 0; k < kAndragogyDimensions; ++k) {
    const double ok = clamp01(o.assumption[k]), dk = clamp01(d.feature[k]);
    t[k] = p.mainEffect[k] * dk + p.interaction[k] * (2.0 * ok - 1.0) * (2.0 * dk - 1.0);
  }
  return t;
}

double engagement(const LearnerOrientation& o, const AndragogicalDesign& d, const AndragogyParams& p) {
  const auto t = engagementTerms(o, d, p);
  return logistic(p.intercept + std::accumulate(t.begin(), t.end(), 0.0));
}

const char* andragogyStepName(AndragogyStep s) {
  static const char* names[kAndragogySteps] = {"set a cooperative learning climate", "create mechanisms for input",
                                               "diagnose learner needs and interests", "formulate learning objectives",
                                               "design sequential activities", "execute the design",
                                               "evaluate and rediagnose needs"};
  const int k = static_cast<int>(s);
  return k >= 0 && k < kAndragogySteps ? names[k] : "?";
}

NeedsCycleRun simulateNeedsCycle(const NeedsCycleParams& p, std::uint64_t seed) {
  if (p.learners < 1 || p.competencies < 1 || p.cycles < 0 || p.objectivesPerCycle < 1 || p.objectivesPerCycle > p.competencies)
    throw std::invalid_argument("simulateNeedsCycle: bad sizes");
  NeedsCycleRun run;
  run.meanGap.assign(p.cycles + 1, 0.0);
  run.hitRate.assign(p.cycles, 0.0);
  // Learner input and a candid climate both reduce the diagnosis error (they compound).
  const double noise = p.diagnosisNoise * (1.0 - 0.6 * clamp01(p.learnerInput)) * (1.0 - 0.4 * clamp01(p.climate));
  const int m = p.competencies, k = p.objectivesPerCycle;
  for (int i = 0; i < p.learners; ++i) {
    Rng rng = Rng::stream(seed, static_cast<std::uint64_t>(i));
    std::vector<double> gap(m);
    for (auto& g : gap) g = std::max(0.0, rng.normal(1.0, 0.6));
    run.meanGap[0] += mean(gap) / p.learners;
    for (int c = 0; c < p.cycles; ++c) {
      std::vector<double> diagnosed(m);
      for (int j = 0; j < m; ++j) diagnosed[j] = gap[j] + noise * rng.normal();
      std::vector<int> byDiagnosis(m), byTruth(m);
      std::iota(byDiagnosis.begin(), byDiagnosis.end(), 0);
      std::iota(byTruth.begin(), byTruth.end(), 0);
      std::sort(byDiagnosis.begin(), byDiagnosis.end(), [&](int a, int b) { return diagnosed[a] > diagnosed[b]; });
      std::sort(byTruth.begin(), byTruth.end(), [&](int a, int b) { return gap[a] > gap[b]; });
      int hits = 0;
      for (int a = 0; a < k; ++a)
        for (int b = 0; b < k; ++b) hits += byDiagnosis[a] == byTruth[b] ? 1 : 0;
      run.hitRate[c] += static_cast<double>(hits) / k / p.learners;
      for (int a = 0; a < k; ++a) gap[byDiagnosis[a]] *= 1.0 - p.gainPerObjective;
      run.meanGap[c + 1] += mean(gap) / p.learners;
    }
  }
  run.finalGap = run.meanGap.back();
  return run;
}

}  // namespace syn
