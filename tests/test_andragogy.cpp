#include <cmath>
#include <numeric>

#include "syn/andragogy/andragogy.hpp"
#include "syn/core/math.hpp"
#include "test_framework.hpp"

using namespace syn;

TEST(andragogy_design_suits_adults_pedagogy_suits_children) {
  const auto child = LearnerOrientation::child(), adult = LearnerOrientation::adult();
  const auto ped = AndragogicalDesign::pedagogical(), andr = AndragogicalDesign::andragogical();
  CHECK(engagement(adult, andr) > engagement(adult, ped) + 0.3);
  CHECK(engagement(child, ped) > engagement(child, andr) + 0.2);
  // The interaction: the andragogical advantage grows with the learner's adult orientation.
  const auto teen = LearnerOrientation::adolescent();
  const double dChild = engagement(child, andr) - engagement(child, ped);
  const double dTeen = engagement(teen, andr) - engagement(teen, ped);
  const double dAdult = engagement(adult, andr) - engagement(adult, ped);
  CHECK(dChild < dTeen && dTeen < dAdult);
}

TEST(engagement_terms_add_up) {
  const auto o = LearnerOrientation::adolescent();
  const auto d = AndragogicalDesign::andragogical();
  AndragogyParams p;
  const auto t = engagementTerms(o, d, p);
  CHECK_NEAR(engagement(o, d, p), logistic(p.intercept + std::accumulate(t.begin(), t.end(), 0.0)), 1e-12);
  // Transparency helps everybody (main effect): a child gains from it too.
  AndragogicalDesign opaque = AndragogicalDesign::pedagogical(), clear = opaque;
  opaque.feature[5] = 0.0;
  clear.feature[5] = 0.5;
  CHECK(engagement(LearnerOrientation::child(), clear) > engagement(LearnerOrientation::child(), opaque));
  CHECK(std::string(andragogyStepName(AndragogyStep::DiagnoseNeeds)) == "diagnose learner needs and interests");
}

TEST(learner_input_sharpens_the_needs_diagnosis) {
  NeedsCycleParams closed, open;
  closed.learnerInput = 0.0;
  closed.climate = 0.0;
  open.learnerInput = 1.0;
  open.climate = 1.0;
  const auto a = simulateNeedsCycle(closed, 2), b = simulateNeedsCycle(open, 2);
  CHECK(b.hitRate[0] > a.hitRate[0] + 0.2);
  CHECK(b.finalGap < a.finalGap);
  for (std::size_t c = 1; c < b.meanGap.size(); ++c) CHECK(b.meanGap[c] < b.meanGap[c - 1]);
  NeedsCycleParams perfect = open;
  perfect.diagnosisNoise = 0.0;
  CHECK_NEAR(simulateNeedsCycle(perfect, 2).hitRate[0], 1.0, 1e-12);
}
