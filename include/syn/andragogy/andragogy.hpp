#pragma once

// Andragogy (Knowles): how adults learn, and how instruction should be set up for them.
//
// Knowles' six assumptions describe the learner; each pairs with one feature of the lesson
// design that serves it:
//   self-concept: dependent → self-directed      ↔ learner input into goals and content
//   reservoir of experience                       ↔ use of learners' experience as a resource
//   readiness tied to social roles                ↔ relevance to roles (employee, parent...)
//   orientation: subject → performance centred    ↔ immediate, competency-based application
//   internal motivation                           ↔ intrinsic appeal (instead of external rewards)
//   need to know why                              ↔ transparency about goals and their purpose
// Engagement is σ(β₀ + Σ_k [m_k·d_k + w_k·(2o_k − 1)(2d_k − 1)]): each design feature d_k has a
// small main effect m_k (transparency helps everybody) and an interaction with how strongly the
// learner holds the matching assumption o_k. Children (low o_k) do well with structured,
// teacher-directed design; adults (high o_k) disengage from it.
//
// Bartle's seven steps turn the assumptions into a process: set a cooperative climate, create
// mechanisms for input, diagnose needs, formulate objectives, design and execute sequential
// activities, then evaluate and rediagnose. The needs-diagnosis cycle is simulated: learners
// have competency gaps, instruction targets the gaps it diagnoses, and learner input together
// with a cooperative (honest) climate makes the diagnosis more accurate.

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace syn {

constexpr int kAndragogyDimensions = 6;

struct LearnerOrientation {
  std::array<double, kAndragogyDimensions> assumption{};  // o_k in [0, 1]
  static LearnerOrientation child();
  static LearnerOrientation adolescent();
  static LearnerOrientation adult();
};

struct AndragogicalDesign {
  std::array<double, kAndragogyDimensions> feature{};  // d_k in [0, 1]
  /** Teacher-directed, subject-centred, rewards-based lesson. */
  static AndragogicalDesign pedagogical();
  /** Self-directed, experience-based, role-relevant, transparent lesson. */
  static AndragogicalDesign andragogical();
};

const char* assumptionName(int k);
const char* designFeatureName(int k);

struct AndragogyParams {
  double intercept = 0.4;
  std::array<double, kAndragogyDimensions> mainEffect = {0.05, 0.05, 0.1, 0.05, 0.0, 0.25};
  std::array<double, kAndragogyDimensions> interaction = {0.45, 0.3, 0.35, 0.35, 0.35, 0.25};
};

double engagement(const LearnerOrientation& o, const AndragogicalDesign& d, const AndragogyParams& p = {});

/** Per-dimension contributions to the engagement logit (main effect + interaction). */
std::array<double, kAndragogyDimensions> engagementTerms(const LearnerOrientation& o, const AndragogicalDesign& d,
                                                         const AndragogyParams& p = {});

enum class AndragogyStep {
  CooperativeClimate = 0, MechanismsForInput, DiagnoseNeeds, FormulateObjectives, DesignActivities, ExecuteDesign, EvaluateAndRediagnose
};
constexpr int kAndragogySteps = 7;
const char* andragogyStepName(AndragogyStep s);

struct NeedsCycleParams {
  int learners = 300;
  int competencies = 8;
  int cycles = 6;
  int objectivesPerCycle = 2;
  double climate = 0.5;         // cooperative climate: how candidly needs are shared, [0, 1]
  double learnerInput = 0.5;    // mechanisms for input, [0, 1]
  double diagnosisNoise = 1.0;  // sd of the diagnosis error with no input and no climate
  double gainPerObjective = 0.45;  // share of a targeted gap closed in a cycle
};

struct NeedsCycleRun {
  std::vector<double> meanGap;    // mean competency gap per cycle (index 0 = before)
  std::vector<double> hitRate;    // share of objectives that target a true top gap, per cycle
  double finalGap = 0.0;
};

NeedsCycleRun simulateNeedsCycle(const NeedsCycleParams& params, std::uint64_t seed);

}  // namespace syn
