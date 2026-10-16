#pragma once

// Vygotsky's Zone of Proximal Development.
//
// A learner of ability θ attempts a task of difficulty b (both on a logit scale, as in item
// response theory). Alone they succeed with P_alone = σ(θ − b); with a more knowledgeable other
// providing scaffolding s they succeed with P_assisted = σ(θ + s − b). The three concentric
// zones are:
//   independent: P_alone ≥ p_hi                          (can already do it alone)
//   proximal:    not independent, and P_assisted ≥ p_lo  (can do it with help)
//   beyond:      P_assisted < p_lo                       (cannot do it yet, even with help)
// Learning from an attempt needs both success (with help) and novelty (not yet mastered
// alone): Δθ = η·P_assisted·(1 − P_alone), which peaks inside the ZPD and vanishes in the
// other two zones (boredom and frustration).

#include <cstdint>
#include <string>
#include <vector>

namespace syn {

enum class Zone { Independent = 0, Proximal, Beyond };
const char* zoneName(Zone z);

struct ZpdParams {
  double independentThreshold = 0.8;  // p_hi
  double assistedThreshold = 0.5;     // p_lo
  double scaffold = 1.5;              // s, logit boost from a more knowledgeable other
  double learningRate = 0.35;         // η
};

double successAlone(double theta, double b);
double successAssisted(double theta, double b, double scaffold);
Zone zoneOf(double theta, double b, const ZpdParams& p);
double expectedGain(double theta, double b, const ZpdParams& p);
/** The band of difficulties [lo, hi] that form the ZPD of a learner of ability θ. */
std::pair<double, double> zpdBand(double theta, const ZpdParams& p);

enum class TaskPolicy { TooEasy, ClassAverage, TooHard, Zpd };
const char* taskPolicyName(TaskPolicy p);

struct TutoringParams {
  int learners = 300;
  int sessions = 30;
  double abilityMean = 0.0;
  double abilitySd = 1.0;
  double classDifficulty = 0.5;   // difficulty used by the ClassAverage policy at session 0
  double curriculumPace = 0.08;   // the class difficulty rises this much per session
  double assessmentNoise = 0.4;   // sd of the tutor's estimate of θ (ZPD policy)
  bool fadeScaffold = true;       // withdraw support as tasks are mastered
};

struct TutoringRun {
  std::string policy;
  std::vector<double> meanAbility;    // per session (before the session's task)
  std::vector<double> q10Ability, q90Ability;
  std::vector<std::vector<double>> zoneShare;  // [zone][session]
  double finalGain = 0.0;             // mean θ_end − θ_start
};

TutoringRun simulateTutoring(const ZpdParams& zpd, const TutoringParams& params, TaskPolicy policy, std::uint64_t seed);

}  // namespace syn
