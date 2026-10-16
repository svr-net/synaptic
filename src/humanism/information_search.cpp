#include "syn/humanism/information_search.hpp"

#include <algorithm>
#include <stdexcept>

#include "syn/core/math.hpp"
#include "syn/core/random.hpp"

namespace syn {

const std::array<IspStageInfo, kIspStages>& ispStages() {
  static const std::array<IspStageInfo, kIspStages> stages = {{
      {"initiation", "uncertainty, apprehension", 0.60, 2.0},
      {"selection", "optimism", 0.40, 2.0},
      {"exploration", "confusion, frustration, doubt", 0.85, 8.0},
      {"formulation", "clarity", 0.35, 3.0},
      {"collection", "sense of direction, confidence", 0.25, 6.0},
      {"presentation", "relief; satisfaction or disappointment", 0.15, 3.0},
  }};
  return stages;
}

IspResult simulateInformationSearch(const IspParams& p, std::uint64_t seed) {
  if (p.students < 1 || p.deadline < 1) throw std::invalid_argument("simulateInformationSearch: bad sizes");
  const auto& stages = ispStages();
  const int horizon = p.deadline + p.deadline / 2;  // keep following late students past the deadline
  IspResult r;
  for (int d = 0; d <= horizon; ++d) r.days.push_back(d);
  r.meanUncertainty.assign(r.days.size(), 0.0);
  for (auto& s : r.stageShare) s.assign(r.days.size(), 0.0);
  std::vector<int> working(r.days.size(), 0);

  const double support = clamp01(p.support);
  double completionDays = 0.0;
  int completed = 0;
  for (int i = 0; i < p.students; ++i) {
    Rng rng = Rng::stream(seed, static_cast<std::uint64_t>(i));
    double anxiety = clamp01(rng.normal(p.anxietyMean, p.anxietySd));
    const double effort = std::max(0.2, rng.normal(p.effortMean, p.effortSd)) * (1.0 + 0.3 * clamp01(p.relevance));
    int stage = 0;
    double progress = 0.0;
    int formulationDay = -1, doneDay = -1;
    for (std::size_t d = 0; d < r.days.size(); ++d) {
      if (stage < kIspStages) {
        const auto& info = stages[stage];
        // Trait anxiety amplifies the stage's typical feeling; acknowledging that the feeling
        // is normal halves it at full support.
        const double felt = clamp01(info.uncertainty * (1.0 + p.anxietyAmplification * (2.0 * anxiety - 1.0)) * (1.0 - 0.5 * support));
        r.meanUncertainty[d] += felt;
        ++working[d];
        r.stageShare[stage][d] += 1.0;
        r.meanDaysInStage[stage] += 1.0;
        const double today = std::max(0.0, effort * (1.0 + 0.25 * rng.normal()));
        progress += today / (1.0 + p.interference * felt) / info.work;
        anxiety = clamp01(anxiety - 0.03 * support * anxiety);
        if (progress >= 1.0) {
          progress = 0.0;
          ++stage;
          if (stage == static_cast<int>(IspStage::Formulation)) formulationDay = static_cast<int>(d) + 1;
          if (stage == kIspStages) doneDay = static_cast<int>(d) + 1;
        }
      } else {
        r.stageShare[kIspStages][d] += 1.0;
        (void)rng.normal();  // keep the random stream aligned across students' paths
      }
    }
    if (doneDay >= 0 && doneDay <= p.deadline) {
      r.completedOnTime += 1.0;
      if (formulationDay >= 0 && formulationDay <= 0.6 * p.deadline) r.satisfied += 1.0;
    }
    if (doneDay >= 0) {
      completionDays += doneDay;
      ++completed;
    }
  }
  const double n = p.students;
  for (std::size_t d = 0; d < r.days.size(); ++d) {
    r.meanUncertainty[d] = working[d] ? r.meanUncertainty[d] / working[d] : 0.0;
    for (auto& s : r.stageShare) s[d] /= n;
  }
  for (auto& m : r.meanDaysInStage) m /= n;
  r.completedOnTime /= n;
  r.satisfied /= n;
  r.meanCompletionDay = completed ? completionDays / completed : 0.0;
  return r;
}

}  // namespace syn
