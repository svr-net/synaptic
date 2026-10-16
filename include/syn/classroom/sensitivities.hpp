#pragma once

// Sensitivity of the learning outcome to each lever: where would a little more of one teaching
// practice help most?
//
// ∂f/∂x_k is estimated by bump-and-rerun, (f(x + h·e_k) − f(x − h·e_k)) / 2h, one-sided at the
// ends of [0, 1]. With common random numbers the two runs share learners and luck, so their
// difference measures the lever, not the noise. With independent seeds the noise of each run
// is divided by 2h and swamps small effects. Repeating the estimate over several base seeds
// gives each method's standard error.

#include <cstdint>
#include <vector>

#include "syn/classroom/classroom_engine.hpp"

namespace syn {

struct LeverSensitivity {
  Lever lever;
  double value = 0.0;
  double crn = 0.0, crnStdError = 0.0;              // common random numbers
  double independent = 0.0, independentStdError = 0.0;  // independent seeds
};

struct SensitivityResult {
  double score = 0.0;
  double bump = 0.0;
  int replications = 0;
  std::vector<LeverSensitivity> levers;
  double meanVarianceReduction = 0.0;  // mean of (se_independent / se_crn)²
};

SensitivityResult leverSensitivities(const ClassroomEngine& engine, const InstructionalDesign& design, double bump,
                                     int replications);

}  // namespace syn
