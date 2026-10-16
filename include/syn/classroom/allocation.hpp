#pragma once

// Allocating a design's learning gain to its levers.
//
// A design improves the delayed test score over a baseline design (by default the plain
// "banking" lecture). The improvement f(design) − f(baseline) is split among the levers that
// differ, in three ways:
//   Shapley:     the average marginal contribution of switching the lever from its baseline to
//                its design value, over random orders of switching. Each order adds up exactly
//                to the total, so the contributions do too.
//   incremental: f(design) − f(design with that lever at baseline): what removing the lever
//                would cost. Interactions make these not add up.
//   standalone:  f(baseline with only that lever changed) − f(baseline).
// All runs share the learners and their random numbers (common random numbers).

#include <cstdint>
#include <vector>

#include "syn/classroom/classroom_engine.hpp"

namespace syn {

struct LeverContribution {
  Lever lever;
  double baselineValue = 0.0, designValue = 0.0;
  double shapley = 0.0, incremental = 0.0, standalone = 0.0;
};

struct AllocationResult {
  double baselineScore = 0.0, designScore = 0.0;
  double totalGain = 0.0;      // designScore − baselineScore
  double shapleySum = 0.0;     // equals totalGain up to rounding
  std::vector<LeverContribution> levers;  // only the levers that differ
  int permutations = 0, evaluations = 0;
};

AllocationResult allocateLearningGain(const ClassroomEngine& engine, const InstructionalDesign& design,
                                      const InstructionalDesign& baseline, int permutations, std::uint64_t orderSeed = 7);

}  // namespace syn
