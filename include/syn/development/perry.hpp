#pragma once

// Perry's scheme of intellectual and ethical development: dualism → multiplicity →
// relativism → commitment.
//
// Development is modelled as a Markov chain over terms: a student in position k moves to k+1
// in a term with probability p_k, raised by instruction that challenges them to compare and
// evaluate conflicting sources (for example, explaining how science measures climate).
//
// The position shapes how a student judges sources. A dualist trusts authority, a student in
// multiplicity treats every view as equally valid and goes by personal experience, a
// relativist weighs evidence by the methods of the discipline, and commitment integrates the
// judgement into their own knowledge. Each position is a set of weights on (authority cue,
// personal resonance, evidence quality); the accuracy of the resulting credibility ranking is
// measured against the true evidence quality.

#include <array>
#include <cstdint>
#include <vector>

namespace syn {

enum class PerryPosition { Dualism = 0, Multiplicity, Relativism, Commitment };
constexpr int kPerryPositions = 4;
const char* perryName(PerryPosition p);

struct PerryParams {
  std::array<double, 3> transition = {0.12, 0.08, 0.05};  // p_k per term, k = dualism..relativism
  double challenge = 0.0;            // instruction that challenges students to evaluate sources, [0, 1]
  double challengeEffect = 1.5;      // multiplies p_k by (1 + effect·challenge) for k ≥ multiplicity
};

/** Distribution over positions after each term (index 0 = start). */
std::vector<std::array<double, kPerryPositions>> perryProgression(const std::array<double, kPerryPositions>& start,
                                                                  const PerryParams& params, int terms);

struct SourceJudgementWeights {
  double authority, personal, evidence;
};
SourceJudgementWeights judgementWeights(PerryPosition p);

/**
 * Accuracy of credibility judgements: share of source pairs a student in the given position
 * orders the same way as their true evidence quality. Authority cues are correlated with
 * quality (`authorityValidity`); personal resonance is not.
 */
double sourceEvaluationAccuracy(PerryPosition p, int sources, int trials, double authorityValidity, std::uint64_t seed);

}  // namespace syn
