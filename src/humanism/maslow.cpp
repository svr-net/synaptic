#include "syn/humanism/maslow.hpp"

#include <stdexcept>

#include "syn/core/math.hpp"

namespace syn {

const char* needName(Need n) {
  switch (n) {
    case Need::Physiological: return "physiological";
    case Need::Safety: return "safety";
    case Need::Belonging: return "belonging and love";
    case Need::Esteem: return "esteem";
    case Need::SelfActualisation: return "self-actualisation";
  }
  return "?";
}

namespace {
double gate(double s, const MaslowParams& p) {
  if (p.width <= 0.0) throw std::invalid_argument("MaslowParams: width must be positive");
  return logistic((s - p.threshold) / p.width);
}
}  // namespace

std::array<double, kNeedLevels> needActivation(const NeedState& s, const MaslowParams& p) {
  std::array<double, kNeedLevels> a{};
  double product = 1.0;
  for (int k = 0; k < kNeedLevels; ++k) {
    a[k] = product;
    product *= gate(s.satisfaction[k], p);
  }
  return a;
}

Need dominantNeed(const NeedState& s, const MaslowParams& p) {
  for (int k = 0; k < kNeedLevels; ++k)
    if (s.satisfaction[k] < p.threshold) return static_cast<Need>(k);
  return Need::SelfActualisation;
}

double learningReadiness(const NeedState& s, const MaslowParams& p) {
  return needActivation(s, p)[static_cast<int>(Need::Esteem)];
}

NeedState applyClassroomClimate(NeedState s, double empathy, double belonging, double recognition) {
  auto lift = [](double& x, double lever) { x += clamp01(lever) * (1.0 - x); };
  lift(s.satisfaction[static_cast<int>(Need::Safety)], empathy);
  lift(s.satisfaction[static_cast<int>(Need::Belonging)], belonging);
  lift(s.satisfaction[static_cast<int>(Need::Esteem)], recognition);
  return s;
}

}  // namespace syn
