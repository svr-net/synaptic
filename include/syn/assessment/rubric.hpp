#pragma once

// Rubrics and self-assessment (chapter 9).
//
// Rubrics. Several instructors rate the same pieces of work (a research paper's sources) on a
// few levels. Each work has a true quality q; a rater perceives q + their severity + noise and
// places it against their own reading of the level boundaries:
//   no rubric       "I know it when I see it": large severity and noise, private boundaries;
//   vague rubric    "some", "often", "rarely": the boundaries move from rater to rater;
//   quantified      counts what is countable (number of scholarly references): raters agree
//                   almost perfectly, but the count says little about quality, and learners who
//                   see the rubric add references, not quality;
//   precise rubric  criteria and levels described in clear terms: raters agree, and learners
//                   who see it know what good work looks like and do better work.
// Agreement is the exact agreement and Cohen's κ averaged over pairs of raters; validity is the
// correlation of one rater's levels with the true quality, averaged over raters.
//
// Self-assessment. People are not good judges of their own abilities, and the least skilled
// overestimate most: perceived = a + β·(0.6 − a) + flattery + noise. Reflecting on one's
// learning helps a little; feedback on the self-assessment, comparing it with what the
// instructor observed, shrinks β and the flattery and helps learning more (Ross 2006).

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace syn {

enum class RubricKind { None = 0, Vague, Quantified, Precise };
const char* rubricKindName(RubricKind k);

struct RubricParams {
  int works = 300;
  int raters = 4;
  int levels = 3;           // novice, intermediate, advanced
  RubricKind kind = RubricKind::Precise;
  bool shared = false;      // learners see the rubric before doing the work
};

struct RubricResult {
  std::string kind;
  double exactAgreement = 0.0;  // pairs of raters giving the same level
  double kappa = 0.0;           // Cohen's κ, mean over pairs
  double validity = 0.0;        // corr(one rater's levels, true quality), mean over raters
  double meanQuality = 0.0;     // of the work (rises when a useful rubric is shared)
  double meanCount = 0.0;       // references per work
};

RubricResult simulateRubric(const RubricParams& params, std::uint64_t seed);

struct SelfAssessmentParams {
  int learners = 400;
  int rounds = 4;                 // assessments with reflection
  bool feedback = true;           // the instructor responds to each self-assessment
  double regression = 0.6;        // β at the start
  double flattery = 0.05;
  double noise = 0.08;
  double feedbackShrink = 0.45;   // share of β and flattery removed by each round of feedback
  double reflection = 0.03;       // skill gained per round by reflecting (doubled with feedback)
};

struct SelfAssessmentResult {
  std::vector<std::array<double, 4>> actual, perceived;  // per round, by quartile of actual skill
  std::vector<double> calibrationError;                  // per round, mean |perceived − actual|
  std::vector<double> meanSkill;                         // per round
};

SelfAssessmentResult simulateSelfAssessment(const SelfAssessmentParams& params, std::uint64_t seed);

}  // namespace syn
