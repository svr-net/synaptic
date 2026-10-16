#pragma once

// Dweck's mindset theory: a fixed mindset treats ability as innate, a growth mindset treats it
// as the outcome of effort, strategy and feedback.
//
// Each learner has a mindset g ∈ [0, 1] (0 fixed, 1 growth), a skill, an effort level and a
// current study strategy of some quality. Each week they choose a challenging or a safe task
// (growth-minded learners seek challenge), succeed with P = σ(skill + effort − difficulty) and
// learn Δskill = η·effort·quality·(challenge ? 1.6 : 0.6).
//   after a failure, growth: keep or raise effort, try a different strategy, use the feedback
//                    fixed:  lower effort; with low effort the learner gives up on the subject
// Instructor feedback shifts the mindset: praising ability ("you're smart") nudges g down;
// praising process and saying "not yet" after a failure nudges g up. Lowering standards makes
// tasks easier, which raises success but removes the challenge that drives learning.

#include <cstdint>
#include <string>
#include <vector>

namespace syn {

enum class FeedbackStyle { Ability, Neutral, ProcessNotYet };
const char* feedbackStyleName(FeedbackStyle f);

struct MindsetParams {
  int learners = 400;
  int weeks = 30;
  double growthMean = 0.5;
  double growthSd = 0.25;
  FeedbackStyle feedback = FeedbackStyle::Neutral;
  bool highStandards = true;
  double coaching = 0.0;     // explicit mindset coaching per week, [0, 1]
  double learningRate = 0.12;
  int strategies = 4;
};

struct MindsetRun {
  std::vector<double> skillFixed, skillGrowth;  // mean skill gain, by starting mindset (< or ≥ 0.5)
  std::vector<double> persisting;               // share still engaged
  std::vector<double> mindset;                  // mean g
  std::vector<double> challengeSeeking;         // share choosing the challenging task
  double finalSkillGain = 0.0;
  double finalPersisting = 0.0;
};

MindsetRun simulateMindset(const MindsetParams& params, std::uint64_t seed);

}  // namespace syn
