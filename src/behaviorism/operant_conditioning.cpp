#include "syn/behaviorism/operant_conditioning.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <stdexcept>

#include "syn/core/math.hpp"
#include "syn/core/random.hpp"

namespace syn {

std::string ReinforcementSchedule::label() const {
  std::ostringstream os;
  switch (type) {
    case ScheduleType::Continuous: return "continuous (CRF)";
    case ScheduleType::FixedRatio: os << "fixed ratio FR" << parameter; break;
    case ScheduleType::VariableRatio: os << "variable ratio VR" << parameter; break;
    case ScheduleType::FixedInterval: os << "fixed interval FI" << parameter; break;
    case ScheduleType::VariableInterval: os << "variable interval VI" << parameter; break;
  }
  return os.str();
}

OperantLearner::OperantLearner(const OperantParams& params) : params_(params), p_(params.baseline) {
  if (!(params.baseline >= 0.0 && params.baseline < 1.0)) throw std::invalid_argument("OperantParams: baseline must lie in [0, 1)");
}

void OperantLearner::reinforce() {
  p_ += params_.acquisitionRate * (1.0 - p_);
  responses_ += 1.0;
  reinforcers_ += 1.0;
  run_ = 0;
}

void OperantLearner::punish() {
  p_ -= params_.punishmentRate * p_;
  responses_ += 1.0;
  run_ = 0;
}

void OperantLearner::unreinforced() {
  const double rHat = expectancy();
  ++run_;
  const double surprise = std::pow(1.0 - std::pow(1.0 - rHat, run_), params_.surpriseExponent);
  p_ -= params_.extinctionRate * surprise * (p_ - params_.baseline);
  responses_ += 1.0;
}

namespace {

// Decides whether a response at opportunity t is reinforced and advances the schedule state.
class ScheduleState {
 public:
  ScheduleState(const ReinforcementSchedule& s, Rng& rng) : s_(s) { reset(rng, 0); }

  bool respond(int t, Rng& rng) {
    bool reinforced = false;
    switch (s_.type) {
      case ScheduleType::Continuous: reinforced = true; break;
      case ScheduleType::FixedRatio:
      case ScheduleType::VariableRatio:
        reinforced = ++count_ >= requirement_;
        break;
      case ScheduleType::FixedInterval:
      case ScheduleType::VariableInterval:
        reinforced = t >= availableAt_;
        break;
    }
    if (reinforced) reset(rng, t);
    return reinforced;
  }

 private:
  void reset(Rng& rng, int t) {
    count_ = 0;
    const double n = std::max(1.0, s_.parameter);
    switch (s_.type) {
      case ScheduleType::Continuous: break;
      case ScheduleType::FixedRatio: requirement_ = static_cast<int>(std::lround(n)); break;
      case ScheduleType::VariableRatio: requirement_ = 1 + rng.uniformInt(static_cast<int>(std::lround(2 * n - 1))); break;
      case ScheduleType::FixedInterval: availableAt_ = t + static_cast<int>(std::lround(n)); break;
      case ScheduleType::VariableInterval: availableAt_ = t + 1 + rng.uniformInt(static_cast<int>(std::lround(2 * n - 1))); break;
    }
  }
  ReinforcementSchedule s_;
  int count_ = 0, requirement_ = 1, availableAt_ = 0;
};

}  // namespace

OperantRun simulateOperant(const OperantParams& params, const ReinforcementSchedule& schedule, int trainingSteps,
                           int extinctionSteps, int learners, std::uint64_t seed) {
  if (trainingSteps < 1 || extinctionSteps < 0 || learners < 1) throw std::invalid_argument("simulateOperant: bad sizes");
  const int steps = trainingSteps + extinctionSteps;
  OperantRun run;
  run.schedule = schedule.label();
  run.trainingSteps = trainingSteps;
  run.probability.assign(steps, 0.0);
  run.cumulativeResponses.assign(steps, 0.0);
  run.cumulativeReinforcers.assign(steps, 0.0);
  double responses = 0.0, reinforcers = 0.0;
  for (int i = 0; i < learners; ++i) {
    Rng rng = Rng::stream(seed, static_cast<std::uint64_t>(i));
    OperantLearner learner(params);
    ScheduleState state(schedule, rng);
    double cumR = 0.0, cumS = 0.0, peak = 0.0;
    int acquired = trainingSteps, extinct = extinctionSteps;
    for (int t = 0; t < steps; ++t) {
      const bool training = t < trainingSteps;
      if (rng.bernoulli(learner.probability())) {
        cumR += 1.0;
        if (training && state.respond(t, rng)) {
          learner.reinforce();
          cumS += 1.0;
        } else {
          learner.unreinforced();
        }
      }
      const double p = learner.probability();
      if (training) {
        peak = std::max(peak, p);
        if (acquired == trainingSteps && p >= 0.8) acquired = t + 1;
      } else if (extinct == extinctionSteps && p < 0.5 * peak) {
        extinct = t - trainingSteps + 1;
      }
      run.probability[t] += p;
      run.cumulativeResponses[t] += cumR;
      run.cumulativeReinforcers[t] += cumS;
    }
    run.acquisitionSteps += acquired;
    run.resistance += extinct;
    responses += cumR;
    reinforcers += cumS;
  }
  const double n = learners;
  for (int t = 0; t < steps; ++t) {
    run.probability[t] /= n;
    run.cumulativeResponses[t] /= n;
    run.cumulativeReinforcers[t] /= n;
  }
  run.acquisitionSteps /= n;
  run.resistance /= n;
  run.reinforcersPerResponse = responses > 0 ? reinforcers / responses : 0.0;
  return run;
}

}  // namespace syn
