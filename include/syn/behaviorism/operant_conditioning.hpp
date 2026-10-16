#pragma once

// Operant conditioning (Skinner): behaviour followed by reinforcement is strengthened,
// behaviour followed by punishment is weakened, and unreinforced behaviour extinguishes.
//
// Each learner emits the target behaviour with probability p on every opportunity. The
// response strength follows a linear-operator (Bush–Mosteller) rule:
//   reinforced:   p ← p + a·(1 − p)
//   punished:     p ← p − c·p
//   unreinforced: p ← p − e·S·(p − p₀),   S = (1 − (1 − r̂)^n)^k
// where p₀ is the operant (baseline) level, r̂ the learner's estimate of how often a response
// pays off (reinforcers per response so far) and n the current run of unreinforced responses.
// S is the surprise of that run: how unlikely it would be under the schedule learned so far.
// After continuous reinforcement one missing reward is already surprising; after an
// intermittent schedule long runs without reward are normal, so the change to extinction is
// noticed late and the behaviour persists (the partial-reinforcement extinction effect).

#include <cstdint>
#include <string>
#include <vector>

namespace syn {

enum class ScheduleType { Continuous, FixedRatio, VariableRatio, FixedInterval, VariableInterval };

struct ReinforcementSchedule {
  ScheduleType type = ScheduleType::Continuous;
  double parameter = 1.0;  // responses per reinforcer (ratio) or opportunities between reinforcers (interval)

  static ReinforcementSchedule continuous() { return {ScheduleType::Continuous, 1.0}; }
  static ReinforcementSchedule fixedRatio(double n) { return {ScheduleType::FixedRatio, n}; }
  static ReinforcementSchedule variableRatio(double n) { return {ScheduleType::VariableRatio, n}; }
  static ReinforcementSchedule fixedInterval(double t) { return {ScheduleType::FixedInterval, t}; }
  static ReinforcementSchedule variableInterval(double t) { return {ScheduleType::VariableInterval, t}; }
  std::string label() const;
};

struct OperantParams {
  double baseline = 0.05;        // p₀: operant level before training
  double acquisitionRate = 0.12; // a
  double extinctionRate = 0.15;  // e
  double punishmentRate = 0.3;   // c
  double surpriseExponent = 6.0; // k: only clearly improbable runs count as a change
};

class OperantLearner {
 public:
  explicit OperantLearner(const OperantParams& params);
  double probability() const { return p_; }
  /** r̂: reinforcers per response so far (with a uniform prior). */
  double expectancy() const { return (reinforcers_ + 1.0) / (responses_ + 2.0); }
  void reinforce();
  void punish();
  void unreinforced();

 private:
  OperantParams params_;
  double p_;
  double responses_ = 0.0, reinforcers_ = 0.0;
  int run_ = 0;
};

struct OperantRun {
  std::string schedule;
  int trainingSteps = 0;
  std::vector<double> probability;            // mean response probability per opportunity
  std::vector<double> cumulativeResponses;    // mean cumulative record (Skinner's recorder)
  std::vector<double> cumulativeReinforcers;  // mean cumulative reinforcers
  double acquisitionSteps = 0.0;  // mean opportunities until p first reaches 0.8 (or training length)
  double resistance = 0.0;        // mean extinction opportunities until p falls below half its peak
  double reinforcersPerResponse = 0.0;
};

/** Trains `learners` learners on the schedule, then runs an extinction phase without reinforcement. */
OperantRun simulateOperant(const OperantParams& params, const ReinforcementSchedule& schedule, int trainingSteps,
                           int extinctionSteps, int learners, std::uint64_t seed);

}  // namespace syn
