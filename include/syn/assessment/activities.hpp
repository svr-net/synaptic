#pragma once

// Assessing learning (chapter 9): assessment activities and what their data say.
//
// Assessment measures progress toward the learning outcomes in order to improve teaching and
// learning; it is not grading (grades mix in attendance, effort and conduct) and not evaluation
// (learners' satisfaction with a session). Activities are summative (at the end) or formative
// (during the session), direct (learners show what they can do) or indirect (they report what
// they think they learned), formal (planned, reach everyone, hard data) or informal (a quick
// check-in), traditional (fixed-choice recall) or authentic (a task in a realistic setting).
//
// Selecting an activity: the first consideration is alignment with the outcome (an activity
// measures one level of Bloom's taxonomy, see outcomes/bloom.hpp: evidence below the outcome's
// level overstates achievement); then the information it yields (noise, reach), the audience
// (complicated writing for young learners or English-language learners, a quiz for adults at a
// public-library workshop) and the time (not 15–20 minutes of a 45-minute one-shot).

#include <cstdint>
#include <string>
#include <vector>

#include "syn/outcomes/bloom.hpp"

namespace syn {

struct AssessmentActivity {
  std::string name;
  double minutes = 5.0;        // class time
  bool direct = true;          // learners demonstrate (false: self-report)
  bool formal = true;          // planned, reaches everyone, hard data
  bool authentic = false;      // a realistic task rather than fixed-choice recall
  BloomLevel measures = BloomLevel::Remember;  // the level its evidence speaks to
  double noise = 0.2;          // measurement error of one learner's result (mastery units)
  double reach = 1.0;          // share of learners it yields data for
  double writingLoad = 0.0;    // demands on writing, [0, 1]
  double testLike = 0.0;       // feels like a test, [0, 1]
  bool afterSession = false;   // completed after class (no class time; feedback later)

  static std::vector<AssessmentActivity> catalogue();
};

/** An audience's tolerance of writing and of test-like activities ([0, 1]). */
struct AssessmentAudience {
  std::string name;
  double writing = 0.7;
  double tests = 0.6;
  static std::vector<AssessmentAudience> all();  // undergraduates, high school, older adults, English learners
};

struct ActivityScore {
  std::string activity;
  double alignment = 0.0;    // evidence level vs outcome level (indirect: low)
  double information = 0.0;  // reach · (1 − noise)
  double fit = 0.0;          // suits the audience
  double timeCost = 0.0;     // share of the session it takes
  double value = 0.0;        // alignment · information · fit − time penalty
};

/** Scores every activity in the catalogue for an outcome, an audience and a session length. */
std::vector<ActivityScore> scoreActivities(BloomLevel outcome, const AssessmentAudience& audience, double sessionMinutes);

/**
 * Assessment, grades, self-report and evaluation as signals of learning. Each learner has true
 * learning ℓ; a direct assessment measures ℓ with noise; a grade mixes ℓ with attendance, effort
 * and conduct; self-report regresses to a flattering mean (the least skilled overestimate most);
 * satisfaction mostly reflects enjoyment. Correlation of each with ℓ.
 */
struct SignalValidity {
  double direct = 0.0, grade = 0.0, selfReport = 0.0, satisfaction = 0.0;
};
SignalValidity simulateSignals(int learners, std::uint64_t seed);

}  // namespace syn
