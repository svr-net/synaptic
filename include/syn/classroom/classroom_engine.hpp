#pragma once

// The classroom engine: every theory of the chapter acting on the same simulated learners.
//
// A course is a sequence of sessions (one content unit each) followed by a delayed test. For
// each learner and session the engine applies, in order:
//   1. needs (Maslow): safety, belonging and esteem, raised by the classroom climate, gate the
//      learner's readiness for growth-oriented learning;
//   2. affect (humanism, Kuhlthau): anxiety interferes with cognitive work, empathy relieves it,
//      unsupported disequilibrium and failure raise it;
//   3. motivation (Svinicki, andragogy, behaviourism): value from interest, relevance, choice,
//      transparency and (for younger learners) rewards, times self-efficacy, gives effort;
//   4. challenge (Vygotsky): the session's task sits at the curriculum level, or at each
//      learner's ZPD under scaffolding; support from the teacher and peers lifts assisted
//      success; ability grows by η·effort·P_assisted·(1 − P_alone);
//   5. development (Piaget): abstract content needs formal operations;
//   6. encoding and memory (cognitivism): the unit is encoded with a depth set by prior
//      activation, problem-first struggle, relevance and dialogue, then practised in three
//      review slots that are spaced or massed, retrieval or restudy, with or without feedback;
//   7. culture (Bourdieu): a mismatch between home and school culture taxes learning unless
//      teaching is inclusive;
//   8. after the session: self-efficacy, anxiety and mindset (Dweck) respond to success or
//      failure, and fixed-mindset learners who fail with low effort may disengage.
// The test score at day t combines what is still retrievable of every unit, the ability to
// tell related problems apart (interleaving) and skill on the final level of the curriculum.
//
// Every learner draws from its own random stream with a fixed number of draws per session, so
// two designs run on the same learners and the same luck (common random numbers): differences
// between designs are not drowned by Monte Carlo noise.

#include <cstdint>
#include <string>
#include <vector>

#include "syn/classroom/instructional_design.hpp"
#include "syn/classroom/learner.hpp"
#include "syn/cognitivism/memory.hpp"
#include "syn/core/math.hpp"

namespace syn {

struct CourseSpec {
  int sessions = 8;
  double sessionGap = 7.0;        // days between sessions
  double difficulty = 0.3;        // curriculum level of the first unit (logit vs. mean ability)
  double difficultyStep = 0.1;    // the curriculum rises this much per session
  double delayedTest = 30.0;      // days after the last session
  double abstraction = 0.6;       // how abstract the content is, [0, 1] (Boolean logic is fairly abstract)
  double atRiskBelow = 0.3;       // a test score below this is "at risk"
  double reportStep = 1.0;        // days between reported points
};

struct SimulationSettings {
  int learners = 500;
  std::uint64_t seed = 2024;
};

/** Distribution of a quantity across learners on a grid of days (like an exposure profile). */
struct LearningProfile {
  std::vector<double> days;
  std::vector<double> mean, q10, q50, q90;
  std::vector<double> atRisk;  // share of learners below the at-risk threshold
};

/** Population averages after each session. */
struct SessionTrace {
  std::vector<double> day;
  std::vector<double> ability, effort, motivation, value, efficacy, anxiety, readiness, growth, engaged, zpdShare;
};

struct LearnerOutcome {
  double endScore = 0.0, delayedScore = 0.0, abilityGain = 0.0, capital = 0.0;
  bool engagedAtEnd = true;
};

struct ClassroomResult {
  LearningProfile score;   // test score over time
  LearningProfile recall;  // retrievable share of the content over time
  SessionTrace sessions;
  std::vector<LearnerOutcome> learners;
  double endScore = 0.0;       // mean score the day after the last session
  double delayedScore = 0.0;   // mean score at the delayed test
  double retention = 0.0;      // delayed / end
  double equityGap = 0.0;      // delayed score: above- minus below-median cultural capital
  double atRiskDelayed = 0.0;  // share at risk at the delayed test
  double abilityGain = 0.0;
  double lastSessionDay = 0.0, delayedTestDay = 0.0;
  int learnerCount = 0;
};

class ClassroomEngine {
 public:
  ClassroomEngine(PopulationSpec population, CourseSpec course, SimulationSettings settings, MemoryParams memory = {});

  ClassroomResult run(const InstructionalDesign& design) const;
  /** Run with another seed (used for independent-seed comparisons). */
  ClassroomResult run(const InstructionalDesign& design, std::uint64_t seed) const;
  /** Mean delayed test score only: the objective that allocation and sensitivities use. */
  double objective(const InstructionalDesign& design) const;
  double objective(const InstructionalDesign& design, std::uint64_t seed) const;

  const PopulationSpec& population() const { return population_; }
  const CourseSpec& course() const { return course_; }
  const SimulationSettings& settings() const { return settings_; }

 private:
  ClassroomResult simulate(const InstructionalDesign& design, std::uint64_t seed, bool profiles) const;
  PopulationSpec population_;
  CourseSpec course_;
  SimulationSettings settings_;
  MemoryModel memory_;
};

}  // namespace syn
