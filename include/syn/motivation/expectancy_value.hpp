#pragma once

// Svinicki's model of motivation: motivation balances the perceived value of the learning
// against the learner's expectation of success (self-efficacy), M = V·E.
//
// The instructor's five levers from the chapter act on V and E:
//   relevance          → raises V
//   appropriate challenge → keeps the task in the band where it is neither boring (low V) nor
//                        overwhelming (low success, so E falls)
//   choice and control → raises V (a stake in the class) and E (control over risk)
//   set up for success → clear expectations and meaningful feedback: E learns faster from
//                        each outcome and success becomes more likely
//   guided self-assessment → shrinks the gap between perceived and actual ability, so the
//                        learner's predictions of success become accurate
// Weekly: effort = M; P(success) = σ(ability + effort − difficulty); ability grows with effort;
// self-efficacy moves towards the outcome (Bandura's mastery experiences).

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace syn {

struct MotivationLevers {
  double relevance = 0.0;
  double challengeFit = 0.0;
  double choice = 0.0;
  double successSetup = 0.0;
  double selfAssessment = 0.0;
  static constexpr int kCount = 5;
  double& operator[](int k);
  double operator[](int k) const;
  static const char* name(int k);
};

struct ExpectancyValueParams {
  int learners = 400;
  int weeks = 15;
  double baseValue = 0.45;
  double efficacyMean = 0.5;
  double efficacySd = 0.15;
  double abilityMean = 0.0;
  double abilitySd = 1.0;
  double biasSd = 0.25;        // sd of the error in perceived ability
  double dropoutBelow = 0.08;  // learners whose motivation stays below this for 3 weeks stop engaging
};

struct TermRun {
  std::vector<double> motivation, value, efficacy, ability, calibrationError, engaged;  // per week
  double finalAbilityGain = 0.0;
  double finalMotivation = 0.0;
  double finalEngaged = 0.0;
};

/** Value of a task at challenge offset (difficulty − ability): an inverted U peaking at a stretch. */
double challengeInterest(double offset);

TermRun simulateTerm(const MotivationLevers& levers, const ExpectancyValueParams& params, std::uint64_t seed);

}  // namespace syn
