#pragma once

// Long-term memory as cognitivists see it: knowledge is encoded, stored and must be retrieved.
//
// A memory trace has a stability S: the number of days after which recall has fallen to 90%.
// Retrievability follows a power-law forgetting curve (Ebbinghaus' curve has a long tail),
//   R(t) = (1 + (19/81)·(t − t_last)/S)^(−1/2),   so R = 0.9 at t − t_last = S.
// Each study event changes S:
//   restudy (rereading, drilling):  S ← S·(1 + g_r + a_r·(1 − R))
//   retrieval that succeeds:        S ← S·(1 + g_q + a_q·(1 − R))      (a_q ≫ a_r)
//   retrieval that fails:           with feedback the answer is relearned, as a restudy of a
//                                   weakened trace: S ← max(S_min, f·S)·(1 + g_r + a_r·(1 − R));
//                                   without feedback nothing changes.
// Gains grow with 1 − R: recalling something that was becoming hard to recall strengthens it
// most (desirable difficulty). That one rule produces the testing effect (retrieval beats
// rereading) and the spacing effect (spaced reviews beat massed ones), the two findings Brown,
// Roediger and McDaniel (2014) build Make It Stick on.

#include <cstdint>
#include <string>
#include <vector>

namespace syn {

enum class StudyMode { Restudy, Retrieval, RetrievalWithFeedback };

const char* studyModeName(StudyMode m);

struct MemoryParams {
  double initialStability = 1.0;  // days, at unit encoding depth
  double restudyFloor = 0.05;     // g_r
  double restudyGain = 0.6;       // a_r
  double retrievalFloor = 0.30;   // g_q
  double retrievalGain = 5.0;     // a_q
  double failureFactor = 0.7;     // f
  double minStability = 0.5;      // S_min
  double pretestBoost = 0.5;      // extra depth from attempting a problem before instruction
};

struct MemoryTrace {
  double stability = 1.0;
  double lastStudy = 0.0;
  int reviews = 0;
  int successes = 0;
  double retrievability(double t) const;
};

class MemoryModel {
 public:
  explicit MemoryModel(MemoryParams params = {});
  const MemoryParams& params() const { return params_; }

  /** Encodes a new item at day t. depth > 0 scales initial stability (elaboration, prior links). */
  MemoryTrace encode(double t, double depth = 1.0) const;

  /**
   * A study event at day t. For retrieval, `u` is a uniform draw in (0, 1) deciding success
   * (success iff u < R), so callers can use common random numbers. Returns true on success
   * (always true for restudy).
   */
  bool study(MemoryTrace& trace, double t, StudyMode mode, double u) const;

  /** Stability multiplier of a successful event at retrievability R. */
  double gain(StudyMode mode, double r) const;

 private:
  MemoryParams params_;
};

/** One study event of a plan: the first event of every plan is the initial encoding at day 0. */
struct StudyEvent {
  double day = 0.0;
  StudyMode mode = StudyMode::Retrieval;
};

struct StudyPlan {
  std::string name;
  std::vector<StudyEvent> reviews;  // after the initial encoding at day 0
  double encodingDepth = 1.0;
};

/** count reviews crammed together (minutes apart) on `day`. */
StudyPlan massedPlan(StudyMode mode, int count, double day);
/** count reviews every `gap` days, starting `gap` days after encoding. */
StudyPlan spacedPlan(StudyMode mode, int count, double gap);
/** Expanding gaps firstGap, firstGap·ratio, ... */
StudyPlan expandingPlan(StudyMode mode, int count, double firstGap, double ratio);

struct RetentionResult {
  std::string name;
  std::vector<double> days;
  std::vector<double> meanRecall;  // expected retrievability per day, over learners
  std::vector<double> q10Recall, q90Recall;
  double recallAtTest = 0.0;       // mean R at the test day
  double meanStabilityAtTest = 0.0;
  double reviewSuccessRate = 0.0;  // share of retrieval attempts that succeeded
};

/**
 * Simulates the plan for `learners` learners whose initial stability is log-normally spread
 * (sd `heterogeneity` in log space) and reports retention up to `horizon` days.
 */
RetentionResult simulatePlan(const MemoryModel& model, const StudyPlan& plan, double testDay, double horizon,
                             int learners, double heterogeneity, std::uint64_t seed);

}  // namespace syn
