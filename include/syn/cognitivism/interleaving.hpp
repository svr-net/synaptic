#pragma once

// Interleaving: practising related problem types mixed together rather than in blocks.
//
// Solving a problem needs two things: choosing the right procedure (discrimination) and
// executing it. Execution skill e_k of type k improves with every practice problem of that
// type: e_k ← e_k + η·(1 − e_k). Discrimination d improves only when the learner actually has
// to choose, with feedback: d ← d + δ. In blocked practice the block tells the learner which
// procedure to use, so they choose only at the start of each block; in interleaved practice
// they choose on every problem. P(correct choice) = 1/K + (1 − 1/K)·(1 − e^(−d)).
// Blocked practice therefore looks better during practice and worse on a mixed test.

#include <cstdint>
#include <vector>

namespace syn {

struct InterleavingParams {
  int categories = 3;            // K related problem types (e.g. AND / OR / NOT searches)
  int problemsPerCategory = 12;
  double initialExecution = 0.3;
  double executionRate = 0.12;   // η
  double discriminationRate = 0.12;  // δ
  double heterogeneity = 0.3;    // spread of the learning rates across learners (log-normal sd)
  int learners = 400;
};

struct PracticeOutcome {
  std::vector<double> practiceAccuracy;  // mean accuracy on each practice problem
  double practiceMean = 0.0;
  double testAccuracy = 0.0;             // mixed, uncued test after practice
  double choiceAccuracy = 0.0;           // P(correct procedure) at test
  double execution = 0.0;                // mean execution skill at test
};

struct InterleavingResult {
  PracticeOutcome blocked, interleaved;
};

InterleavingResult simulateInterleaving(const InterleavingParams& params, std::uint64_t seed);

}  // namespace syn
