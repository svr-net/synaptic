#include <algorithm>
#include <cmath>
#include <string>

#include "syn/assessment/activities.hpp"
#include "syn/assessment/rubric.hpp"
#include "syn/assessment/session.hpp"
#include "test_framework.hpp"

using namespace syn;

namespace {
std::string best(BloomLevel outcome, const AssessmentAudience& audience, double minutes = 45.0) {
  auto s = scoreActivities(outcome, audience, minutes);
  return std::max_element(s.begin(), s.end(), [](const auto& a, const auto& b) { return a.value < b.value; })->activity;
}
double valueOf(BloomLevel outcome, const AssessmentAudience& audience, const std::string& name, double minutes = 45.0) {
  for (const auto& s : scoreActivities(outcome, audience, minutes))
    if (s.activity == name) return s.value;
  return -1e9;
}
}  // namespace

TEST(select_assessment_by_outcome_audience_and_time) {
  const auto audiences = AssessmentAudience::all();
  const auto& undergrads = audiences[0];
  const auto& olderAdults = audiences[2];
  const auto& englishLearners = audiences[3];
  // Creating search strings tells more than naming the operators (an "apply" outcome).
  CHECK(best(BloomLevel::Apply, undergrads) == "worksheet");
  CHECK(valueOf(BloomLevel::Apply, undergrads, "worksheet") > valueOf(BloomLevel::Apply, undergrads, "pre/post test (fixed choice)") + 0.3);
  // Indirect evidence: learners tell, they do not show.
  CHECK(valueOf(BloomLevel::Apply, undergrads, "self-assessment questionnaire") < 0.3);
  // A quiz may suit a school, not adults at a public-library workshop.
  CHECK(valueOf(BloomLevel::Remember, audiences[1], "pre/post test (fixed choice)") >
        valueOf(BloomLevel::Remember, olderAdults, "pre/post test (fixed choice)") + 0.3);
  // Complicated writing does not suit English-language learners.
  CHECK(valueOf(BloomLevel::Understand, englishLearners, "directed paraphrasing") <
        valueOf(BloomLevel::Understand, undergrads, "directed paraphrasing"));
  // Evaluating sources: the annotated bibliography (after class, no class time).
  CHECK(best(BloomLevel::Evaluate, undergrads) == "annotated bibliography");
  // Time: the same activity costs more of a shorter session.
  CHECK(valueOf(BloomLevel::Apply, undergrads, "worksheet", 30.0) < valueOf(BloomLevel::Apply, undergrads, "worksheet", 60.0));
  CHECK_THROWS(scoreActivities(BloomLevel::Apply, undergrads, 0.0));
  CHECK(AssessmentActivity::catalogue().size() >= 12);
}

TEST(assessment_grades_self_report_and_satisfaction) {
  const auto v = simulateSignals(2000, 3);
  CHECK(v.direct > v.grade);
  CHECK(v.grade > v.satisfaction);
  CHECK(v.direct > v.selfReport + 0.2);
  CHECK(v.satisfaction < 0.4);
}

TEST(formative_assessment_fixes_gaps_during_the_session) {
  AssessedSessionParams p;
  const auto none = simulateAssessedSession(p, 7);
  p.formative = FormativeCheck::Informal;
  const auto informal = simulateAssessedSession(p, 7);
  p.formative = FormativeCheck::Formal;
  const auto formal = simulateAssessedSession(p, 7);
  CHECK_NEAR(none.reteaches, 0.0, 1e-12);
  CHECK_NEAR(none.assessmentMinutes, 0.0, 1e-12);
  CHECK(informal.outcomeMet > none.outcomeMet + 0.03);
  CHECK(formal.outcomeMet > none.outcomeMet + 0.02);
  // Informal checks are quick but flattering: they miss gaps a formal check finds.
  CHECK(informal.missedGaps > formal.missedGaps + 0.3);
  CHECK(formal.assessmentMinutes > informal.assessmentMinutes);
  // Without a summative assessment the instructor does not know whether the outcome was met.
  p.formative = FormativeCheck::None;
  p.summative = true;
  const auto summative = simulateAssessedSession(p, 7);
  CHECK(summative.estimateError < 0.1);
  CHECK(none.estimateError > 0.15);
  CHECK(summative.outcomeMet <= none.outcomeMet + 1e-12);  // it helps no one in the room
  // Chunks build on each other.
  for (std::size_t j = 1; j < none.chunkMastery.size(); ++j) CHECK(none.chunkMastery[j] < none.chunkMastery[j - 1]);
}

TEST(the_assessment_cycle_improves_teaching) {
  AssessmentCycleParams cp;
  cp.session.classes = 200;
  cp.session.chunkClarity = {0.4, -1.2, 0.3, -0.5};  // one part of the lesson is unclear
  auto run = [&](CycleEvidence e) {
    cp.evidence = e;
    return simulateAssessmentCycle(cp, 7);
  };
  const auto none = run(CycleEvidence::None), grades = run(CycleEvidence::Grades), self = run(CycleEvidence::SelfReport),
             direct = run(CycleEvidence::Direct);
  const auto last = [](const AssessmentCycleResult& r) { return r.met.back(); };
  CHECK(last(direct) > direct.met.front() + 0.1);   // gaps found and fixed
  CHECK(direct.clarity.back() > direct.clarity.front());
  CHECK(last(direct) > last(none) + 0.1);
  CHECK(last(direct) > last(self) + 0.1);            // self-report hides the gaps
  CHECK(last(direct) > last(grades) + 0.1);          // grades are not about the outcome
  CHECK_NEAR(none.clarity.back(), none.clarity.front(), 1e-12);
  // An outcome most learners meet becomes more challenging.
  cp.session.chunkClarity = {1.5, 1.5, 1.5, 1.5};
  const auto easy = run(CycleEvidence::Direct);
  CHECK(easy.difficulty.back() > easy.difficulty.front());
  CHECK_THROWS(simulateAssessmentCycle(AssessmentCycleParams{0}, 1));
}

TEST(rubrics_agreement_and_validity) {
  auto run = [](RubricKind k, bool shared = false, int levels = 3) {
    RubricParams p;
    p.kind = k;
    p.shared = shared;
    p.levels = levels;
    return simulateRubric(p, 7);
  };
  const auto none = run(RubricKind::None), vague = run(RubricKind::Vague), counted = run(RubricKind::Quantified),
             precise = run(RubricKind::Precise);
  // "I know it when I see it" vs criteria and levels described in clear terms.
  CHECK(precise.kappa > vague.kappa + 0.1);
  CHECK(vague.kappa > none.kappa);
  CHECK(precise.validity > none.validity);
  // Counting references: perfect agreement about the wrong thing.
  CHECK(counted.kappa > 0.95);
  CHECK(counted.validity < 0.5);
  const auto countedShared = run(RubricKind::Quantified, true);
  CHECK(countedShared.meanCount > counted.meanCount + 1.0);          // quantity
  CHECK_NEAR(countedShared.meanQuality, counted.meanQuality, 1e-12);  // not quality
  // A shared precise rubric tells learners what good work looks like.
  CHECK(run(RubricKind::Precise, true).meanQuality > precise.meanQuality + 0.2);
  // More levels: harder to agree on.
  CHECK(run(RubricKind::Precise, false, 6).exactAgreement < run(RubricKind::Precise, false, 3).exactAgreement);
  CHECK_THROWS(run(RubricKind::Precise, false, 1));
}

TEST(self_assessment_and_feedback) {
  SelfAssessmentParams p;
  p.feedback = false;
  const auto alone = simulateSelfAssessment(p, 7);
  p.feedback = true;
  const auto withFeedback = simulateSelfAssessment(p, 7);
  // The least skilled overestimate most; the most skilled underestimate.
  const auto& a = alone.actual[0];
  const auto& s = alone.perceived[0];
  CHECK(s[0] - a[0] > 0.2);
  CHECK(s[3] < a[3]);
  CHECK(s[0] - a[0] > s[1] - a[1]);
  // Feedback on self-assessments calibrates them and helps learning.
  CHECK(withFeedback.calibrationError.back() < 0.6 * alone.calibrationError.back());
  CHECK(withFeedback.meanSkill.back() > alone.meanSkill.back());
  CHECK(alone.meanSkill.back() > alone.meanSkill.front());  // reflection alone helps a little
  CHECK_NEAR(alone.calibrationError[0], withFeedback.calibrationError[0], 1e-12);
}
