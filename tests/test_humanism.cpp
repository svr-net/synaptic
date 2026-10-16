#include "syn/humanism/information_search.hpp"
#include "syn/humanism/maslow.hpp"
#include "test_framework.hpp"

using namespace syn;

TEST(maslow_needs_are_prepotent) {
  NeedState met;
  met.satisfaction = {0.95, 0.9, 0.9, 0.9, 0.5};
  const auto a = needActivation(met);
  CHECK_NEAR(a[0], 1.0, 1e-12);
  for (int k = 1; k < kNeedLevels; ++k) CHECK(a[k] <= a[k - 1]);
  CHECK(learningReadiness(met) > 0.9);
  CHECK(dominantNeed(met) == Need::SelfActualisation);

  NeedState hungry = met;
  hungry.satisfaction[0] = 0.2;  // basic needs unmet: nothing above them motivates
  CHECK(dominantNeed(hungry) == Need::Physiological);
  CHECK(learningReadiness(hungry) < 0.01);

  NeedState unsafe = met;
  unsafe.satisfaction[1] = 0.3;
  CHECK(dominantNeed(unsafe) == Need::Safety);
  CHECK(learningReadiness(unsafe) < learningReadiness(met));
}

TEST(supportive_climate_raises_readiness) {
  NeedState s;
  s.satisfaction = {0.9, 0.45, 0.4, 0.4, 0.2};
  const double before = learningReadiness(s);
  const NeedState after = applyClassroomClimate(s, 0.8, 0.8, 0.5);
  CHECK(after[Need::Safety] > s[Need::Safety]);
  CHECK(after[Need::Belonging] > s[Need::Belonging]);
  CHECK(after[Need::Physiological] == s[Need::Physiological]);  // the teacher cannot feed the class
  CHECK(learningReadiness(after) > before + 0.5);
  CHECK(std::string(needName(Need::Esteem)) == "esteem");
}

TEST(information_search_process_affect) {
  IspParams p;
  const auto r = simulateInformationSearch(p, 3);
  // Exploration is the stage of most uncertainty and takes longest.
  const auto& st = ispStages();
  for (int k = 0; k < kIspStages; ++k)
    if (k != static_cast<int>(IspStage::Exploration)) {
      CHECK(st[k].uncertainty < st[2].uncertainty);
      CHECK(r.meanDaysInStage[k] < r.meanDaysInStage[2]);
    }
  // Stage shares sum to one every day.
  for (std::size_t d = 0; d < r.days.size(); ++d) {
    double total = 0.0;
    for (const auto& s : r.stageShare) total += s[d];
    CHECK_NEAR(total, 1.0, 1e-9);
  }
  CHECK(r.satisfied <= r.completedOnTime);
}

TEST(empathetic_support_reduces_anxiety_and_helps_completion) {
  IspParams none, full;
  full.support = 1.0;
  const auto a = simulateInformationSearch(none, 7), b = simulateInformationSearch(full, 7);
  CHECK(b.completedOnTime > a.completedOnTime + 0.2);
  CHECK(b.meanUncertainty[5] < a.meanUncertainty[5]);
  IspParams anxious = none;
  anxious.anxietyMean = 0.8;
  CHECK(simulateInformationSearch(anxious, 7).completedOnTime < a.completedOnTime);
}
