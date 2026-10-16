// End-to-end tour of the library: one experiment per theory of the chapter, then the
// chapter's Activity 3.3 (a lesson on Boolean operators) designed by each theory and run on
// the same simulated class, with the learning gain allocated to the teaching practices.

#include <cstdio>

#include "syn/syn.hpp"

using namespace syn;

int main() {
  std::printf("== Behaviourism\n");
  const auto pav = pavlovExperiment();
  std::printf("  Pavlov: bell strength after 30 pairings %.3f, after 30 bells alone %.3f\n", pav.strengths[29][0], pav.strengths.back()[0]);
  const auto blocking = blockingExperiment();
  std::printf("  Kamin blocking: tone %.3f after light pre-training vs %.3f without\n", blocking.blockedTone, blocking.controlTone);
  for (const auto& p : fearConditioningExperiment().probes) std::printf("  Watson: fear of %-14s %.2f\n", p.stimulus.c_str(), p.fear);
  for (auto s : {ReinforcementSchedule::continuous(), ReinforcementSchedule::variableRatio(5)}) {
    const auto r = simulateOperant({}, s, 300, 300, 300, 1);
    std::printf("  Skinner %-18s: %.0f opportunities to acquire, %.0f to extinguish\n", r.schedule.c_str(), r.acquisitionSteps, r.resistance);
  }
  const auto tokens = simulateTokenEconomy({}, 1);
  std::printf("  point system: engagement %.2f before, %.2f with points, %.2f after withdrawal\n", tokens.before, tokens.during, tokens.after);

  std::printf("== Humanism\n");
  for (double support : {0.0, 1.0}) {
    IspParams p;
    p.support = support;
    const auto r = simulateInformationSearch(p, 1);
    std::printf("  research paper, support %.0f: %.0f%% on time, %.1f days exploring\n", support, 100 * r.completedOnTime, r.meanDaysInStage[2]);
  }

  std::printf("== Cognitivism\n");
  MemoryModel memory;
  for (const auto& plan : {massedPlan(StudyMode::Restudy, 4, 0), spacedPlan(StudyMode::RetrievalWithFeedback, 4, 3)}) {
    const auto r = simulatePlan(memory, plan, 30, 30, 500, 0.3, 1);
    std::printf("  %-42s recall at day 30: %.2f\n", plan.name.c_str(), r.recallAtTest);
  }
  const auto il = simulateInterleaving({}, 1);
  std::printf("  blocked practice %.2f / test %.2f; interleaved practice %.2f / test %.2f\n", il.blocked.practiceMean,
              il.blocked.testAccuracy, il.interleaved.practiceMean, il.interleaved.testAccuracy);

  std::printf("== Constructivism\n");
  for (double feedback : {0.0, 0.3}) {
    AnimalWorld w;
    w.feedback = feedback;
    const auto r = simulateEquilibration({}, w, 1);
    std::printf("  dogs and cats, adult names the animal %.0f%% of the time: accuracy %.2f, %.2f accommodations\n", 100 * feedback,
                r.finalAccuracy, r.accommodations);
  }

  std::printf("== Social constructivism and the ZPD\n");
  for (auto policy : {TaskPolicy::TooEasy, TaskPolicy::ClassAverage, TaskPolicy::TooHard, TaskPolicy::Zpd})
    std::printf("  %-34s ability gain %.2f\n", taskPolicyName(policy), simulateTutoring({}, {}, policy, 1).finalGain);
  for (auto g : {Grouping::None, Grouping::MixedAbility}) {
    DialogueParams p;
    p.grouping = g;
    const auto r = simulateDialogue(p, 1);
    std::printf("  %-34s understanding %.2f, gap between home cultures %.3f\n", r.grouping.c_str(), r.finalMean, r.finalGap);
  }

  std::printf("== Developmental stages\n");
  for (const auto& a : climateChangeLesson())
    std::printf("  %-55s fits age 5: %.2f  age 9: %.2f  age 15: %.2f\n", a.name.c_str(), activityFit(a, 5), activityFit(a, 9), activityFit(a, 15));
  for (int k = 0; k < kPerryPositions; ++k)
    std::printf("  Perry %-13s source judgement accuracy %.2f\n", perryName(static_cast<PerryPosition>(k)),
                sourceEvaluationAccuracy(static_cast<PerryPosition>(k), 8, 200, 0.3, 1));

  std::printf("== Andragogy\n");
  std::printf("  engagement: child %.2f (pedagogical) %.2f (andragogical); adult %.2f / %.2f\n",
              engagement(LearnerOrientation::child(), AndragogicalDesign::pedagogical()),
              engagement(LearnerOrientation::child(), AndragogicalDesign::andragogical()),
              engagement(LearnerOrientation::adult(), AndragogicalDesign::pedagogical()),
              engagement(LearnerOrientation::adult(), AndragogicalDesign::andragogical()));

  std::printf("== Motivation and mindset\n");
  MotivationLevers all;
  for (int k = 0; k < MotivationLevers::kCount; ++k) all[k] = 1.0;
  std::printf("  term ability gain: %.2f with no levers, %.2f with all five\n", simulateTerm({}, {}, 1).finalAbilityGain,
              simulateTerm(all, {}, 1).finalAbilityGain);
  for (auto f : {FeedbackStyle::Ability, FeedbackStyle::ProcessNotYet}) {
    MindsetParams p;
    p.feedback = f;
    const auto r = simulateMindset(p, 1);
    std::printf("  %-28s mindset %.2f, still trying %.0f%%\n", feedbackStyleName(f), r.mindset.back(), 100 * r.finalPersisting);
  }

  std::printf("== Activity 3.3: a Boolean-operators lesson designed by each theory\n");
  for (const auto& pop : {PopulationSpec::children(), PopulationSpec::secondary(), PopulationSpec::adults()}) {
    ClassroomEngine engine(pop, CourseSpec{}, SimulationSettings{500, 2024});
    std::printf("  %s\n", pop.name.c_str());
    for (int t = 0; t < kTheories; ++t) {
      const auto r = engine.run(theoryDesign(static_cast<Theory>(t)));
      std::printf("    %-22s end of course %.2f  after a month %.2f  at risk %3.0f%%  culture gap %+.3f\n",
                  theoryName(static_cast<Theory>(t)), r.endScore, r.delayedScore, 100 * r.atRiskDelayed, r.equityGap);
    }
  }

  ClassroomEngine engine(PopulationSpec::secondary(), CourseSpec{}, SimulationSettings{500, 2024});
  const auto alloc = allocateLearningGain(engine, theoryDesign(Theory::Blended), theoryDesign(Theory::Banking), 20);
  std::printf("== Where the blended lesson's gain comes from (%.3f over the banking lecture)\n", alloc.totalGain);
  for (const auto& c : alloc.levers)
    std::printf("  %-20s Shapley %+.3f  incremental %+.3f  standalone %+.3f\n", leverName(c.lever), c.shapley, c.incremental, c.standalone);
  std::printf("  sum of Shapley contributions %.3f\n", alloc.shapleySum);
  return 0;
}
