#include <cmath>

#include "syn/classroom/allocation.hpp"
#include "syn/classroom/classroom_engine.hpp"
#include "syn/classroom/sensitivities.hpp"
#include "test_framework.hpp"

using namespace syn;

namespace {
ClassroomEngine engine(PopulationSpec pop = PopulationSpec::secondary(), int learners = 300) {
  return ClassroomEngine(pop, CourseSpec{}, SimulationSettings{learners, 2024});
}
}  // namespace

TEST(population_sampling) {
  const auto a = samplePopulation(PopulationSpec::adults(), 50, 1), b = samplePopulation(PopulationSpec::adults(), 50, 1);
  CHECK(a.size() == 50);
  for (std::size_t i = 0; i < a.size(); ++i) CHECK(a[i].ability == b[i].ability);
  for (const auto& l : a) CHECK(l.capital >= 0.0 && l.capital <= 1.0 && l.anxiety >= 0.0 && l.anxiety <= 1.0);
  CHECK_NEAR(adultness(9.0), 0.0, 1e-12);
  CHECK_NEAR(adultness(30.0), 1.0, 1e-12);
}

TEST(designs_and_presets) {
  for (int t = 0; t < kTheories; ++t)
    for (double x : theoryDesign(static_cast<Theory>(t)).lever) CHECK(x >= 0.0 && x <= 1.0);
  CHECK(theoryDesign(Theory::Banking)[Lever::TeacherDirection] == 1.0);
  InstructionalDesign wild;
  wild[Lever::Spacing] = 3.0;
  wild[Lever::Choice] = -1.0;
  CHECK(wild.clamped()[Lever::Spacing] == 1.0 && wild.clamped()[Lever::Choice] == 0.0);
  std::array<double, kGagneEvents> all{};
  all.fill(1.0);
  const auto g = designFromGagne(all);
  CHECK(g[Lever::Transparency] == 1.0 && g[Lever::PriorActivation] == 1.0 && g[Lever::Feedback] == 1.0);
  CHECK(g[Lever::Empathy] == 0.0);  // Gagné's events do not address affect
}

TEST(engine_is_reproducible_and_profiles_are_consistent) {
  const auto e = engine();
  const auto d = theoryDesign(Theory::Cognitivism);
  const auto a = e.run(d), b = e.run(d);
  CHECK(a.delayedScore == b.delayedScore);
  CHECK_NEAR(e.objective(d), a.delayedScore, 1e-15);
  CHECK(e.run(d, 99).delayedScore != a.delayedScore);
  const auto& s = a.score;
  for (std::size_t k = 0; k < s.days.size(); ++k) {
    CHECK(s.q10[k] <= s.q50[k] + 1e-12 && s.q50[k] <= s.q90[k] + 1e-12);
    CHECK(s.atRisk[k] >= 0.0 && s.atRisk[k] <= 1.0);
  }
  CHECK_NEAR(s.days.back(), a.delayedTestDay, 1e-12);
  // The course builds the score up: the end-of-course score beats the score after lesson one.
  CHECK(a.endScore > s.mean.front());
  // After the course, scores only fall (forgetting).
  for (std::size_t k = 1; k < s.days.size(); ++k)
    if (s.days[k] > a.lastSessionDay + 0.5) CHECK(s.mean[k] <= s.mean[k - 1] + 1e-12);
  CHECK(a.retention > 0.0 && a.retention < 1.0);
  CHECK(a.learners.size() == 300);
}

TEST(theories_compared_on_the_same_learners) {
  const auto e = engine();
  const double banking = e.objective(theoryDesign(Theory::Banking));
  double bestSingle = 0.0;
  for (int t = 1; t < kTheories - 1; ++t) {
    const double v = e.objective(theoryDesign(static_cast<Theory>(t)));
    CHECK(v > banking);  // every theory improves on the banking lecture
    bestSingle = std::max(bestSingle, v);
  }
  CHECK(e.objective(theoryDesign(Theory::Blended)) > bestSingle);  // the chapter's conclusion
  // Retrieval, spacing and interleaving: cognitivism retains best of the single theories.
  const auto cog = e.run(theoryDesign(Theory::Cognitivism)), hum = e.run(theoryDesign(Theory::Humanism));
  CHECK(cog.retention > hum.retention);
  // Inclusive social constructivism narrows the gap between home cultures.
  CHECK(e.run(theoryDesign(Theory::SocialConstructivism)).equityGap < cog.equityGap);
}

TEST(andragogy_matters_more_for_adults) {
  const auto kids = engine(PopulationSpec::children()), adults = engine(PopulationSpec::adults());
  const auto bank = theoryDesign(Theory::Banking), andr = theoryDesign(Theory::Andragogy);
  CHECK(adults.objective(andr) / adults.objective(bank) > kids.objective(andr) / kids.objective(bank));
  // Rewards work better on children than on adults.
  const auto beh = theoryDesign(Theory::Behaviorism);
  CHECK(kids.objective(beh) / kids.objective(bank) > adults.objective(beh) / adults.objective(bank));
}

TEST(gagne_events_each_contribute) {
  const auto e = engine();
  std::array<double, kGagneEvents> all{};
  all.fill(1.0);
  const double full = e.objective(designFromGagne(all));
  for (int k : {static_cast<int>(GagneEvent::StimulateRecall), static_cast<int>(GagneEvent::ElicitPerformance),
                static_cast<int>(GagneEvent::EnhanceRetention)}) {
    auto missing = all;
    missing[k] = 0.0;
    CHECK(e.objective(designFromGagne(missing)) < full);
  }
}

TEST(shapley_allocation_adds_up) {
  const auto e = engine(PopulationSpec::secondary(), 150);
  const auto r = allocateLearningGain(e, theoryDesign(Theory::Blended), theoryDesign(Theory::Banking), 8);
  CHECK(r.levers.size() == static_cast<std::size_t>(kLevers));  // the blended design changes every lever
  CHECK_NEAR(r.shapleySum, r.totalGain, 1e-12);
  CHECK(r.totalGain > 0.2);
  CHECK_NEAR(r.designScore, e.objective(theoryDesign(Theory::Blended)), 1e-15);
  double standalone = 0.0;
  for (const auto& c : r.levers) standalone += c.standalone;
  CHECK(std::fabs(standalone - r.totalGain) > 0.01);  // interactions: standalone effects do not add up
  // Same design and baseline: nothing to allocate.
  const auto none = allocateLearningGain(e, theoryDesign(Theory::Banking), theoryDesign(Theory::Banking), 3);
  CHECK(none.levers.empty() && none.totalGain == 0.0);
}

TEST(common_random_numbers_sharpen_sensitivities) {
  const auto e = engine(PopulationSpec::secondary(), 120);
  const auto s = leverSensitivities(e, theoryDesign(Theory::Blended), 0.1, 4);
  CHECK(s.levers.size() == static_cast<std::size_t>(kLevers));
  int sharper = 0;
  for (const auto& l : s.levers) sharper += l.crnStdError < l.independentStdError ? 1 : 0;
  CHECK(sharper >= kLevers - 1);
  CHECK(s.meanVarianceReduction > 10.0);
  const auto& spacing = s.levers[static_cast<int>(Lever::Spacing)];
  CHECK(spacing.crn > 0.0);
  CHECK_THROWS(leverSensitivities(e, theoryDesign(Theory::Blended), 0.0, 4));
}
