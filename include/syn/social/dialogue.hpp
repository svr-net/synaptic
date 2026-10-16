#pragma once

// Social constructivism in a classroom: learning through dialogue with peers (Vygotsky) in a
// school whose culture matches some families better than others (Bourdieu).
//
// Each learner has an understanding u ∈ [0, 1] of the topic and a cultural capital c ∈ [0, 1]:
// how closely their home culture matches the culture of the classroom. Per session:
//   instruction:   Δu = η·(1 − κ·(1 − c)·(1 − ι))·(1 − u)
//   peer dialogue: in small groups each learner moves towards more knowledgeable members,
//                  Δu_i += π·mean_j max(0, u_j − u_i)
//   articulation:  explaining to others deepens the explainer's own understanding,
//                  Δu_j += α·mean_i max(0, u_j − u_i)·(1 − u_j)
// κ is the penalty for a mismatch between home and school culture, and ι ∈ [0, 1] how inclusive
// and culturally responsive the teaching is. Starting understanding is correlated with c,
// because schools reflect the culture of wealthier families.

#include <cstdint>
#include <string>
#include <vector>

namespace syn {

enum class Grouping { None, Random, MixedAbility, SameAbility };
const char* groupingName(Grouping g);

struct DialogueParams {
  int learners = 32;
  int groupSize = 4;
  int sessions = 20;
  Grouping grouping = Grouping::MixedAbility;
  double instructionRate = 0.06;   // η
  double peerRate = 0.25;          // π
  double articulationRate = 0.08;  // α
  double capitalPenalty = 0.6;     // κ
  double inclusivity = 0.0;        // ι
  double capitalMean = 0.55;
  double capitalSd = 0.25;
  double priorBase = 0.15;         // initial understanding = priorBase + priorCapital·c + noise
  double priorCapital = 0.2;
  int classes = 50;                // independent classes simulated (Monte Carlo)
};

struct DialogueRun {
  std::string grouping;
  std::vector<double> meanUnderstanding;  // per session (index 0: before teaching)
  std::vector<double> highCapital;        // learners above the median cultural capital
  std::vector<double> lowCapital;         // learners below it
  std::vector<double> spread;             // sd of understanding
  double finalGap = 0.0;                  // high − low at the end
  double finalMean = 0.0;
};

DialogueRun simulateDialogue(const DialogueParams& params, std::uint64_t seed);

}  // namespace syn
