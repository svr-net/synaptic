#include <cmath>

#include "syn/outcomes/backward_design.hpp"
#include "syn/outcomes/bloom.hpp"
#include "syn/social/zpd.hpp"
#include "test_framework.hpp"

using namespace syn;

TEST(table_8_3_poorly_and_strongly_worded_outcomes) {
  // Left column: poorly worded. Right column: strongly worded.
  for (const char* poor : {"Be able to search databases", "Evaluate a website", "Understand how databases work", "Understand primary sources"})
    CHECK(!analyseOutcome(poor).strong);
  for (const char* strong : {"Apply Boolean search strategies effectively to broaden and narrow searches.",
                             "Evaluate websites for authority, reliability, and accuracy; identify trustworthy sites.",
                             "Describe how information is parsed and stored in a database.",
                             "Define primary, secondary, and tertiary sources; identify examples of each."})
    CHECK(analyseOutcome(strong).strong);
  const auto u = analyseOutcome("Understand how databases work");
  CHECK(u.vague);
  CHECK(!u.measurable);
  CHECK(!u.issues.empty());
  const auto b = analyseOutcome("By the end of this session learners will be able to apply Boolean operators to narrow a search");
  CHECK(b.measurable);
  CHECK(b.level == static_cast<int>(BloomLevel::Apply));
  CHECK(b.verb == "apply");
  const auto d = analyseOutcome("Define primary, secondary, and tertiary sources; identify examples of each.");
  CHECK(d.level == static_cast<int>(BloomLevel::Remember));
  CHECK(d.actions == 2);
  CHECK(analyseOutcome("Design a search strategy for a research question").level == static_cast<int>(BloomLevel::Create));
  CHECK(!analyseOutcome("Be familiar with the library catalogue").measurable);
  CHECK(!higherOrder(BloomLevel::Apply));
  CHECK(higherOrder(BloomLevel::Analyze));
  CHECK(std::string(bloomName(BloomLevel::Create)) == "create");
}

TEST(bloom_levels_are_hierarchical_and_evidence_must_match) {
  const auto m = simulateLevelMastery(400, 1.5, 3);
  for (const auto& x : m.learners)
    for (int k = 1; k < kBloomLevels; ++k) CHECK(x[k] <= x[k - 1] + 1e-12);
  // Evidence below the outcome's level overstates achievement; above it, understates it.
  const auto recall = checkEvidence(m, BloomLevel::Apply, BloomLevel::Remember, 0.1, 5);
  const auto match = checkEvidence(m, BloomLevel::Apply, BloomLevel::Apply, 0.1, 5);
  const auto above = checkEvidence(m, BloomLevel::Apply, BloomLevel::Create, 0.1, 5);
  CHECK(recall.bias > 0.2);
  CHECK(std::fabs(match.bias) < 0.03);
  CHECK(above.bias < -0.1);
  CHECK(match.accuracy > recall.accuracy + 0.3);
  CHECK_NEAR(match.actual, recall.actual, 1e-12);
}

TEST(twin_sins_of_coverage_and_activity) {
  const auto items = librarySessionContent();
  SessionParams p;
  const auto coverage = simulateSession(items, coveragePlan(items, 50), p, 7);
  const auto activity = simulateSession(items, activityPlan(items, 50), p, 7);
  const auto backward = simulateSession(items, backwardDesignPlan(items, 50, 4), p, 7);
  // Coverage touches everything and teaches little.
  CHECK_NEAR(coverage.coverage, 1.0, 1e-12);
  CHECK(coverage.enduring < 0.25 * backward.enduring);
  CHECK(coverage.outcomesMet < 0.05);
  // Activities teach basic skills, but without the big idea there is no transfer.
  CHECK(activity.basics > coverage.basics);
  CHECK_NEAR(activity.bigIdeas, 0.0, 1e-12);
  CHECK(backward.transfer > activity.transfer + 0.1);
  // Learners go back to their old habits unless they see when and why.
  CHECK(backward.behaviourChange > activity.behaviourChange + 0.5);
  CHECK(backward.behaviourChange > coverage.behaviourChange + 0.5);
  CHECK(backward.enduring > activity.enduring);
  // Nice-to-know items are not outcomes.
  CHECK_NEAR(backward.niceToKnow, 0.0, 1e-12);
}

TEST(three_to_four_outcomes_in_an_hour) {
  const auto items = librarySessionContent();
  SessionParams p;
  auto best = [&](double minutes) {
    int k = 1;
    double top = -1.0;
    for (int n = 1; n <= 12; ++n) {
      const double e = simulateSession(items, backwardDesignPlan(items, minutes, n), p, 7).enduring;
      if (e > top) {
        top = e;
        k = n;
      }
    }
    return k;
  };
  const int hour = best(60);
  CHECK(hour >= 3 && hour <= 4);
  CHECK(best(30) <= hour);
  CHECK(best(90) >= hour);
}

TEST(priority_builds_basics_before_their_big_idea) {
  const auto items = librarySessionContent();
  const auto order = outcomePriority(items);
  CHECK(order.size() == items.size());
  for (std::size_t a = 0; a < order.size(); ++a)
    if (items[order[a]].kind == ContentKind::Basic)
      for (std::size_t b = 0; b < a; ++b) CHECK(order[b] != items[order[a]].bigIdea);
  CHECK(items[order.back()].kind == ContentKind::NiceToKnow);
}

TEST(shared_outcomes_and_essential_questions) {
  const auto items = librarySessionContent();
  SessionParams p;
  auto plan = backwardDesignPlan(items, 60, 4);
  const auto shared = simulateSession(items, plan, p, 7);
  plan.outcomesShared = false;
  const auto hidden = simulateSession(items, plan, p, 7);
  CHECK(shared.selfAssessment > hidden.selfAssessment + 0.05);  // learners can monitor their learning
  plan.essentialQuestions = 0.0;
  plan.affective = 0.0;
  const auto plain = simulateSession(items, plan, p, 7);
  CHECK(hidden.behaviourChange > plain.behaviourChange);  // relevance made explicit
  CHECK_NEAR(plain.enduring, shared.enduring, 1e-12);     // same teaching, same mastery
}

TEST(outcome_level_appropriate_to_the_audience) {
  int prev = -1;
  for (double theta : {-2.0, -1.0, 0.0, 1.0, 2.0}) {
    const auto f = fitOutcomeLevel(theta, 1.5);
    CHECK(f.recommended >= prev);
    prev = f.recommended;
    for (int k = 1; k < kBloomLevels; ++k) CHECK(f.alone[k] < f.alone[k - 1]);
    for (int k = 0; k < kBloomLevels; ++k) CHECK(f.assisted[k] >= f.alone[k]);
  }
  CHECK(fitOutcomeLevel(-2.0, 1.5).recommended < fitOutcomeLevel(1.0, 1.5).recommended);
  CHECK(fitOutcomeLevel(-2.0, 1.5).zone[1] == static_cast<int>(Zone::Proximal));
}

TEST(outcomes_bad_inputs) {
  const auto items = librarySessionContent();
  CHECK_THROWS(backwardDesignPlan(items, 60, 0));
  SessionPlan bad;
  CHECK_THROWS(simulateSession(items, bad, SessionParams{}, 1));
  CHECK_THROWS(simulateLevelMastery(0, 1.0, 1));
}
