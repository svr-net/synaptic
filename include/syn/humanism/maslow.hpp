#pragma once

// Maslow's hierarchy of needs. Five levels, each with a satisfaction in [0, 1]. The levels are
// prepotent: a higher need only motivates once the needs below it are reasonably satisfied.
// We model this with a smooth gate g(s) = σ((s − threshold)/width): the share of level k's
// motivating potential that is available is the product of the gates of every level below it.

#include <array>
#include <string>

namespace syn {

enum class Need { Physiological = 0, Safety, Belonging, Esteem, SelfActualisation };
constexpr int kNeedLevels = 5;

const char* needName(Need n);

struct NeedState {
  std::array<double, kNeedLevels> satisfaction{};  // [0, 1] per level, bottom first
  double operator[](Need n) const { return satisfaction[static_cast<int>(n)]; }
};

struct MaslowParams {
  double threshold = 0.55;
  double width = 0.07;
};

/** Gate product for each level: how much that level can currently motivate behaviour. */
std::array<double, kNeedLevels> needActivation(const NeedState& s, const MaslowParams& p = {});

/** The lowest level whose satisfaction is still below the threshold (the dominant need). */
Need dominantNeed(const NeedState& s, const MaslowParams& p = {});

/**
 * Readiness for growth-oriented learning: the gate product through belonging, i.e. how far
 * esteem and self-actualisation (learning for its own sake) are within reach.
 */
double learningReadiness(const NeedState& s, const MaslowParams& p = {});

/**
 * The humanist teacher's contribution: an empathetic, supportive classroom raises safety and
 * belonging; recognition of effort and accomplishment raises esteem. Each lever is in [0, 1]
 * and closes that share of the gap to full satisfaction.
 */
NeedState applyClassroomClimate(NeedState s, double empathy, double belonging, double recognition);

}  // namespace syn
