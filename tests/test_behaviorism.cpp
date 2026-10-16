#include "syn/behaviorism/classical_conditioning.hpp"
#include "syn/behaviorism/operant_conditioning.hpp"
#include "syn/behaviorism/token_economy.hpp"
#include "test_framework.hpp"

using namespace syn;

TEST(rescorla_wagner_matches_closed_form) {
  RescorlaWagner rw({{"bell", 0.3}}, 0.5);
  for (int n = 1; n <= 40; ++n) {
    rw.trial({0}, true);
    CHECK_NEAR(rw.strength(0), acquisitionCurve(0.3, 0.5, 1.0, n), 1e-12);
  }
}

TEST(pavlov_acquisition_then_extinction) {
  const auto run = pavlovExperiment(0.3, 30, 30);
  CHECK(run.strengths.size() == 60);
  CHECK(run.phaseStart[1] == 30);
  const double acquired = run.strengths[29][0];
  CHECK(acquired > 0.95);
  for (int t = 1; t < 30; ++t) CHECK(run.strengths[t][0] > run.strengths[t - 1][0]);  // monotone acquisition
  CHECK(run.strengths.back()[0] < 0.1 * acquired);                                    // extinguished
}

TEST(kamin_blocking) {
  const auto r = blockingExperiment();
  CHECK(r.controlTone > 0.4);            // compound alone: the two cues share the prediction
  CHECK(r.blockedTone < 0.1);            // the pre-trained light blocks the tone
  CHECK(r.blocked.strengths.back()[0] > 0.9);
}

TEST(fear_generalises_to_similar_stimuli) {
  const auto r = fearConditioningExperiment();
  CHECK(r.probes.front().stimulus == "white rat");
  CHECK(r.probes.front().fear > 0.9);
  for (std::size_t i = 1; i < r.probes.size(); ++i) CHECK(r.probes[i].fear < r.probes[i - 1].fear);
  CHECK(r.probes.back().fear < 0.06);  // wooden blocks do not frighten
  // Distances are geodesic angles read from the CGA inner product of the stimulus points.
  const double expected[] = {0.0, 0.4, 0.8, 1.0, 1.4, 3.0};
  for (std::size_t i = 0; i < r.probes.size(); ++i) CHECK_NEAR(r.probes[i].distance, expected[i], 1e-9);
}

TEST(operant_reinforcement_and_punishment) {
  OperantParams p;
  OperantLearner a(p);
  for (int i = 0; i < 40; ++i) a.reinforce();
  CHECK(a.probability() > 0.95);
  const double before = a.probability();
  a.punish();
  CHECK_NEAR(a.probability(), before * (1 - p.punishmentRate), 1e-12);
  CHECK_NEAR(a.expectancy(), 41.0 / 43.0, 1e-12);
}

TEST(continuous_reinforcement_is_learned_fastest) {
  OperantParams p;
  const auto crf = simulateOperant(p, ReinforcementSchedule::continuous(), 300, 200, 200, 3);
  const auto fr = simulateOperant(p, ReinforcementSchedule::fixedRatio(5), 300, 200, 200, 3);
  CHECK(crf.acquisitionSteps < fr.acquisitionSteps);
  CHECK(crf.probability[299] > 0.99);
  CHECK(crf.reinforcersPerResponse > 3 * fr.reinforcersPerResponse);
  // Cumulative records never decrease.
  for (std::size_t t = 1; t < fr.cumulativeResponses.size(); ++t) CHECK(fr.cumulativeResponses[t] >= fr.cumulativeResponses[t - 1]);
}

TEST(partial_reinforcement_extinction_effect) {
  OperantParams p;
  const auto crf = simulateOperant(p, ReinforcementSchedule::continuous(), 300, 300, 300, 5);
  for (auto s : {ReinforcementSchedule::fixedRatio(5), ReinforcementSchedule::variableRatio(5), ReinforcementSchedule::variableInterval(5)}) {
    const auto partial = simulateOperant(p, s, 300, 300, 300, 5);
    CHECK(partial.resistance > 1.8 * crf.resistance);
  }
  CHECK(crf.probability.back() < 0.1);  // eventually extinguished
}

TEST(token_economy_raises_engagement_while_it_runs) {
  TokenEconomyParams p;
  const auto r = simulateTokenEconomy(p, 11);
  CHECK(r.during > r.before + 0.15);
  CHECK(r.after < r.during);
  CHECK(r.engagementHigh[0] > r.engagementLow[0]);
  CHECK(r.points.back() > 10.0);
  // Points stop accruing once the system is withdrawn.
  CHECK_NEAR(r.points.back(), r.points[p.rewardEnd - 1], 1e-12);
  // Without undermining or reinforcement the intrinsic interest is unchanged.
  TokenEconomyParams q = p;
  q.undermining = 0.0;
  const auto s = simulateTokenEconomy(q, 11);
  CHECK_NEAR(s.intrinsic.front(), s.intrinsic.back(), 1e-12);
  CHECK(r.intrinsic.back() < s.intrinsic.back());
}
