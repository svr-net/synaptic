#pragma once

// Learning outcomes and Bloom's taxonomy (chapter 8).
//
// The revised taxonomy (Anderson & Krathwohl 2001) has six hierarchical levels: remember,
// understand, apply (lower order), analyse, evaluate, create (higher order). An outcome is
// written as "By the end of this session learners will be able to <verb> <knowledge>"; the verb
// names the cognitive process and should be an observable action. Vague verbs ("understand",
// "know", "be familiar with", "learn") do not say what learners will do, so the outcome cannot
// be measured, and learners may read them as mere recall.
//
// Because the levels are hierarchical, mastery at a level is bounded by mastery of the levels
// below it. An assessment at level L measures mastery at L, so evidence collected below the level
// of the outcome (naming the Boolean operators for an outcome about using them) overstates
// achievement: Backward Design's stage 2 must match the evidence to the outcome.

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace syn {

enum class BloomLevel { Remember = 0, Understand, Apply, Analyze, Evaluate, Create };
constexpr int kBloomLevels = 6;

const char* bloomName(BloomLevel l);
bool higherOrder(BloomLevel l);
/** Action verbs associated with a level (lower case). */
const std::vector<std::string>& bloomVerbs(BloomLevel l);
/** Verbs and phrases to avoid: they do not describe an observable action. */
const std::vector<std::string>& vagueVerbs();

struct OutcomeAnalysis {
  std::string text;
  std::string verb;              // the leading verb or phrase found
  int level = -1;                // Bloom level of the verb, −1 if none
  bool vague = false;            // "understand", "know", "be familiar with"...
  bool measurable = false;       // an action verb at a known level
  bool specific = false;         // names a criterion, purpose or condition, not just an object
  bool strong = false;           // measurable and specific (Table 8.3's right column)
  int actions = 0;               // action verbs found (several: a compound outcome)
  std::vector<std::string> issues;
};

/** Analyses one written outcome (the "learners will be able to" stem is optional). */
OutcomeAnalysis analyseOutcome(const std::string& text);

/**
 * Mastery of one skill at each Bloom level for a population of learners: learning proceeds up
 * the hierarchy, mastery(k) = mastery(k − 1) · u_k with u_k ∈ (0, 1].
 */
struct LevelMastery {
  std::vector<std::array<double, kBloomLevels>> learners;
};
LevelMastery simulateLevelMastery(int learners, double depth, std::uint64_t seed);

/**
 * Evidence at one Bloom level for an outcome at another: mean measured achievement, mean true
 * achievement (at the outcome's level), and the share of learners whose "met / not met"
 * judgement (threshold 0.6) is right.
 */
struct EvidenceCheck {
  BloomLevel outcome, evidence;
  double measured = 0.0, actual = 0.0, bias = 0.0, accuracy = 0.0;
};
EvidenceCheck checkEvidence(const LevelMastery& m, BloomLevel outcome, BloomLevel evidence, double noise, std::uint64_t seed);

}  // namespace syn
