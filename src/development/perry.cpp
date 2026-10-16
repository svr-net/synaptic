#include "syn/development/perry.hpp"

#include <cmath>
#include <stdexcept>

#include "syn/core/math.hpp"
#include "syn/core/random.hpp"

namespace syn {

const char* perryName(PerryPosition p) {
  switch (p) {
    case PerryPosition::Dualism: return "dualism";
    case PerryPosition::Multiplicity: return "multiplicity";
    case PerryPosition::Relativism: return "relativism";
    case PerryPosition::Commitment: return "commitment";
  }
  return "?";
}

std::vector<std::array<double, kPerryPositions>> perryProgression(const std::array<double, kPerryPositions>& start,
                                                                  const PerryParams& params, int terms) {
  if (terms < 0) throw std::invalid_argument("perryProgression: negative terms");
  double total = 0.0;
  for (double x : start) {
    if (x < 0.0) throw std::invalid_argument("perryProgression: negative probability");
    total += x;
  }
  if (std::fabs(total - 1.0) > 1e-9) throw std::invalid_argument("perryProgression: start must sum to 1");
  std::vector<std::array<double, kPerryPositions>> out{start};
  for (int t = 0; t < terms; ++t) {
    const auto& cur = out.back();
    std::array<double, kPerryPositions> next{};
    for (int k = 0; k < kPerryPositions; ++k) {
      double move = 0.0;
      if (k < kPerryPositions - 1) {
        const double boost = k >= 1 ? 1.0 + params.challengeEffect * clamp01(params.challenge) : 1.0;
        move = clamp01(params.transition[k] * boost);
      }
      next[k] += cur[k] * (1.0 - move);
      if (k + 1 < kPerryPositions) next[k + 1] += cur[k] * move;
    }
    out.push_back(next);
  }
  return out;
}

SourceJudgementWeights judgementWeights(PerryPosition p) {
  switch (p) {
    case PerryPosition::Dualism: return {0.85, 0.10, 0.05};
    case PerryPosition::Multiplicity: return {0.10, 0.85, 0.05};
    case PerryPosition::Relativism: return {0.20, 0.10, 0.70};
    case PerryPosition::Commitment: return {0.15, 0.05, 0.80};
  }
  return {1.0 / 3, 1.0 / 3, 1.0 / 3};
}

double sourceEvaluationAccuracy(PerryPosition p, int sources, int trials, double authorityValidity, std::uint64_t seed) {
  if (sources < 2 || trials < 1) throw std::invalid_argument("sourceEvaluationAccuracy: bad sizes");
  const auto w = judgementWeights(p);
  const double rho = clamp(authorityValidity, -1.0, 1.0);
  double total = 0.0;
  for (int t = 0; t < trials; ++t) {
    Rng rng = Rng::stream(seed, static_cast<std::uint64_t>(t));
    std::vector<double> quality(sources), judged(sources);
    for (int s = 0; s < sources; ++s) {
      const double q = rng.normal();
      const double authority = rho * q + std::sqrt(1.0 - rho * rho) * rng.normal();
      const double personal = rng.normal();
      quality[s] = q;
      judged[s] = w.authority * authority + w.personal * personal + w.evidence * q + 0.2 * rng.normal();
    }
    total += pairwiseConcordance(quality, judged);
  }
  return total / trials;
}

}  // namespace syn
