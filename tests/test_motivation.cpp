#include "syn/motivation/expectancy_value.hpp"
#include "syn/motivation/mindset.hpp"
#include "test_framework.hpp"

using namespace syn;

TEST(challenge_interest_is_an_inverted_u) {
  CHECK(challengeInterest(0.6) > challengeInterest(-1.5));  // too easy: boring
  CHECK(challengeInterest(0.6) > challengeInterest(3.0));   // too hard: overwhelming
  CHECK(challengeInterest(0.6) >= challengeInterest(0.5));
  CHECK(challengeInterest(0.6) >= challengeInterest(0.7));
}

TEST(each_motivation_lever_helps) {
  ExpectancyValueParams p;
  MotivationLevers none;
  const auto base = simulateTerm(none, p, 3);
  // Relevance, challenge, choice and success set-up raise learning; self-assessment calibrates (below).
  for (int k = 0; k < 4; ++k) {
    MotivationLevers one;
    one[k] = 1.0;
    const auto r = simulateTerm(one, p, 3);
    CHECK(r.finalAbilityGain > base.finalAbilityGain + 0.1);
  }
  MotivationLevers all;
  for (int k = 0; k < MotivationLevers::kCount; ++k) all[k] = 1.0;
  const auto best = simulateTerm(all, p, 3);
  CHECK(best.finalAbilityGain > 2.0 * base.finalAbilityGain);
  CHECK(best.finalMotivation > base.finalMotivation + 0.2);
  CHECK(best.finalEngaged >= base.finalEngaged);
  // Guided self-assessment calibrates learners' view of their own ability.
  MotivationLevers assess;
  assess.selfAssessment = 1.0;
  CHECK(simulateTerm(assess, p, 3).calibrationError.back() < 0.5 * base.calibrationError.back());
  CHECK_THROWS(none[7]);
}

TEST(growth_mindset_persists_and_learns_more) {
  MindsetParams p;
  const auto r = simulateMindset(p, 4);
  CHECK(r.skillGrowth.back() > r.skillFixed.back() + 0.5);
  for (std::size_t t = 1; t < r.persisting.size(); ++t) CHECK(r.persisting[t] <= r.persisting[t - 1] + 1e-12);
}

TEST(not_yet_feedback_shifts_the_mindset) {
  MindsetParams ability, process;
  ability.feedback = FeedbackStyle::Ability;
  process.feedback = FeedbackStyle::ProcessNotYet;
  const auto a = simulateMindset(ability, 4), b = simulateMindset(process, 4);
  CHECK(b.mindset.back() > a.mindset.back() + 0.2);
  CHECK(b.finalPersisting > a.finalPersisting + 0.2);
  CHECK(b.finalSkillGain > a.finalSkillGain);
  // Coaching alone also moves learners towards a growth mindset.
  MindsetParams coached;
  coached.coaching = 1.0;
  CHECK(simulateMindset(coached, 4).mindset.back() > simulateMindset(MindsetParams{}, 4).mindset.back() + 0.2);
  // Lowering standards raises success but removes the stretch that drives learning.
  MindsetParams low = process;
  low.highStandards = false;
  CHECK(simulateMindset(low, 4).finalSkillGain < b.finalSkillGain);
}
