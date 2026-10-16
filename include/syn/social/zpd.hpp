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
//
// Geometry (pure elliptic CGA, see core/elliptic_cga.hpp). The learner is the point L at a pole
// of the circle of tasks, and a task of relative difficulty Δ = b − θ is the point T at angle
// φ = 2·atan(e^{Δ/2}) from L. Then
//   P_alone    = 1 + ½·T·L                       (= cos²(φ/2) = σ(θ − b), exactly)
//   P_assisted = 1 + ½·T'·L,  T' = D T D̃,  D = exp(−s/4 · n e)
// The scaffold is a dilation versor: it pulls every task towards the learner by scaling
// tan(φ/2) by e^{−s/2}, i.e. Δ ↦ Δ − s. The zones are concentric caps around L:
//   independent  T inside the cap of radius ρ_hi,  cos ρ_hi = 2·p_hi − 1
//   proximal     not independent, and T' inside the cap of radius ρ_lo, cos ρ_lo = 2·p_lo − 1
// which is the chapter's Figure 3.2 (three concentric circles) on the sphere.

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

/** One task assessed once in the geometry: both success probabilities and the zone. */
struct ZpdAssessment {
  double alone = 0.0, assisted = 0.0;
  Zone zone = Zone::Independent;
};
ZpdAssessment assessTask(double theta, double b, const ZpdParams& p);

/** Angle φ of a task of relative difficulty Δ = b − θ from the learner: 2·atan(e^{Δ/2}). */
double taskAngle(double relativeDifficulty);

/**
 * Figure 3.2: tasks on the 2-sphere around the learner (at the pole), classified by the zone
 * caps and drawn by stereographic projection from the antipode, where the caps become three
 * concentric circles. Each task is a relative difficulty and an azimuth; the rings are evenly
 * spaced in the plane between the easiest and the hardest task.
 */
struct ZpdMapPoint {
  double x = 0.0, y = 0.0;  // stereographic coordinates (radius = tan(φ/2) = e^{Δ/2})
  double difficulty = 0.0;  // Δ = b − θ
  Zone zone = Zone::Independent;
};
std::vector<ZpdMapPoint> zpdMap(const ZpdParams& p, int rings, int perRing, double minDifficulty, double maxDifficulty);
/** Radii of the two zone boundaries on the stereographic map: tan(ρ_hi/2) and tan(ρ_lo'/2). */
std::pair<double, double> zpdMapRadii(const ZpdParams& p);

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
