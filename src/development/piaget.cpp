#include "syn/development/piaget.hpp"

#include <cmath>
#include <algorithm>
#include <stdexcept>

namespace syn {

const std::array<PiagetStageInfo, kPiagetStages>& piagetStages() {
  static const std::array<PiagetStageInfo, kPiagetStages> s = {{
      {"sensorimotor", "birth to 18-24 months", 0.0},
      {"preoperational", "18-24 months to 7 years", 2.0},
      {"concrete operational", "7 to 12 years", 7.0},
      {"formal operational", "12 years and up", 12.0},
  }};
  return s;
}

const char* capabilityName(Capability c) {
  switch (c) {
    case Capability::Reflexes: return "inherent reflexes";
    case Capability::Language: return "language";
    case Capability::TrialAndError: return "trial-and-error problem solving";
    case Capability::ImaginativePlay: return "imaginative play";
    case Capability::LogicalThinking: return "logical thinking";
    case Capability::AppearanceVsReality: return "appearance is not always reality";
    case Capability::OtherPerspectives: return "other people's perspectives";
    case Capability::AbstractThinking: return "abstract thinking";
    case Capability::DeductiveReasoning: return "deductive reasoning";
    case Capability::Consequences: return "thinking through consequences";
  }
  return "?";
}

bool stageProvides(PiagetStage s, Capability c) {
  static const PiagetStage firstStage[kCapabilities] = {
      PiagetStage::Sensorimotor,          // reflexes
      PiagetStage::Preoperational,        // language
      PiagetStage::Preoperational,        // trial and error
      PiagetStage::Preoperational,        // imaginative play
      PiagetStage::ConcreteOperational,   // logical thinking
      PiagetStage::ConcreteOperational,   // appearance vs reality
      PiagetStage::ConcreteOperational,   // other perspectives
      PiagetStage::FormalOperational,     // abstract thinking
      PiagetStage::FormalOperational,     // deductive reasoning
      PiagetStage::FormalOperational,     // consequences
  };
  return static_cast<int>(s) >= static_cast<int>(firstStage[static_cast<int>(c)]);
}

PiagetStage nominalStage(double age) {
  if (age < 0.0) throw std::invalid_argument("nominalStage: negative age");
  if (age < 2.0) return PiagetStage::Sensorimotor;
  if (age < 7.0) return PiagetStage::Preoperational;
  if (age < 12.0) return PiagetStage::ConcreteOperational;
  return PiagetStage::FormalOperational;
}

std::array<double, kPiagetStages> stageDistribution(double age, double onsetSd) {
  if (onsetSd <= 0.0) throw std::invalid_argument("stageDistribution: onsetSd must be positive");
  // reached[k] = P(onset of stage k ≤ age); the stages are hierarchical, so P(stage = k) is
  // reached[k] − reached[k + 1].
  std::array<double, kPiagetStages + 1> reached{};
  reached[0] = 1.0;
  for (int k = 1; k < kPiagetStages; ++k)
    reached[k] = 0.5 * std::erfc(-(age - piagetStages()[k].nominalOnset) / (onsetSd * std::sqrt(2.0)));
  reached[kPiagetStages] = 0.0;
  std::array<double, kPiagetStages> p{};
  for (int k = 0; k < kPiagetStages; ++k) p[k] = std::max(0.0, reached[k] - reached[k + 1]);
  return p;
}

double activityFit(const LearningActivity& a, double age, double onsetSd) {
  const auto dist = stageDistribution(age, onsetSd);
  double fit = 0.0;
  for (int k = 0; k < kPiagetStages; ++k) {
    bool ok = true;
    for (auto c : a.demands) ok = ok && stageProvides(static_cast<PiagetStage>(k), c);
    if (ok) fit += dist[k];
  }
  return fit;
}

std::vector<LearningActivity> climateChangeLesson() {
  return {
      {"look at pictures of animal habitats", {Capability::Language}},
      {"nature walk: describe what you see", {Capability::Language, Capability::TrialAndError}},
      {"reflect on your own experience of weather", {Capability::Language, Capability::ImaginativePlay}},
      {"imagine how changes affect other people and organisms", {Capability::OtherPerspectives, Capability::LogicalThinking}},
      {"anticipate consequences of climate change", {Capability::AbstractThinking, Capability::Consequences}},
      {"propose steps to improve the environment", {Capability::DeductiveReasoning, Capability::Consequences, Capability::OtherPerspectives}},
  };
}

}  // namespace syn
