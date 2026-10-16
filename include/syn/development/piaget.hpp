#pragma once

// Piaget's four stages of cognitive development (Table 3.1 of the chapter) and the fit of
// learning activities to a learner's stage.
//
// Each stage enables a set of capabilities. Children reach the stages at roughly, but not
// exactly, the nominal ages: we take the age at which stage k begins to be normally
// distributed around its nominal onset (2, 7 and 12 years) with a common spread, which gives
// the probability of each stage at any age. An activity demands certain capabilities; its
// expected fit for a learner is the probability that their stage provides all of them.

#include <array>
#include <string>
#include <vector>

namespace syn {

enum class PiagetStage { Sensorimotor = 0, Preoperational, ConcreteOperational, FormalOperational };
constexpr int kPiagetStages = 4;

enum class Capability {
  Reflexes = 0,          // react to the environment with inherent reflexes
  Language,              // begin to develop language
  TrialAndError,         // basic problem solving by trial and error
  ImaginativePlay,
  LogicalThinking,
  AppearanceVsReality,   // appearance is not always reality
  OtherPerspectives,
  AbstractThinking,
  DeductiveReasoning,
  Consequences,          // think through consequences
};
constexpr int kCapabilities = 10;

struct PiagetStageInfo {
  const char* name;
  const char* ages;
  double nominalOnset;  // years
};

const std::array<PiagetStageInfo, kPiagetStages>& piagetStages();
const char* capabilityName(Capability c);

/** Whether a stage provides a capability (stages are cumulative). */
bool stageProvides(PiagetStage s, Capability c);

/** Stage at the nominal ages: < 2 sensorimotor, < 7 preoperational, < 12 concrete, else formal. */
PiagetStage nominalStage(double ageYears);

/** P(stage = k | age) when stage onsets vary between children with sd `onsetSd` years. */
std::array<double, kPiagetStages> stageDistribution(double ageYears, double onsetSd = 0.9);

struct LearningActivity {
  std::string name;
  std::vector<Capability> demands;
};

/** Expected share of learners of that age for whom the activity is developmentally appropriate. */
double activityFit(const LearningActivity& a, double ageYears, double onsetSd = 0.9);

/** The chapter's climate-change lesson, from a nature walk to proposing solutions. */
std::vector<LearningActivity> climateChangeLesson();

}  // namespace syn
