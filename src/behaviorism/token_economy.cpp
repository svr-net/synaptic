#include "syn/behaviorism/token_economy.hpp"

#include <algorithm>
#include <stdexcept>

#include "syn/core/math.hpp"
#include "syn/core/random.hpp"

namespace syn {

TokenEconomyRun simulateTokenEconomy(const TokenEconomyParams& p, std::uint64_t seed) {
  if (p.students < 2 || p.days < 1 || p.levelPoints < 1) throw std::invalid_argument("simulateTokenEconomy: bad sizes");
  const auto days = static_cast<std::size_t>(p.days);
  TokenEconomyRun run;
  run.engagement.assign(days, 0.0);
  run.engagementLow.assign(days, 0.0);
  run.engagementHigh.assign(days, 0.0);
  run.points.assign(days, 0.0);
  run.intrinsic.assign(days, 0.0);

  std::vector<double> intrinsic(p.students), habit(p.students, 0.0), points(p.students, 0.0);
  std::vector<Rng> rngs;
  for (int i = 0; i < p.students; ++i) {
    rngs.push_back(Rng::stream(seed, static_cast<std::uint64_t>(i)));
    intrinsic[i] = rngs.back().normal(p.intrinsicMean, p.intrinsicSd);
  }
  const double median = quantile(intrinsic, 0.5);
  std::vector<bool> high(p.students);
  int nHigh = 0;
  for (int i = 0; i < p.students; ++i) nHigh += (high[i] = intrinsic[i] > median) ? 1 : 0;
  const int nLow = p.students - nHigh;

  double sums[3] = {0, 0, 0};
  int counts[3] = {0, 0, 0};
  for (int d = 0; d < p.days; ++d) {
    const bool rewards = d >= p.rewardStart && d < p.rewardEnd;
    for (int i = 0; i < p.students; ++i) {
      const double level = static_cast<double>(static_cast<int>(points[i]) / p.levelPoints);
      const double incentive = rewards ? p.incentive + p.levelBonus * level : 0.0;
      const double prob = logistic(p.baseline + intrinsic[i] + habit[i] + incentive);
      run.engagement[d] += prob / p.students;
      (high[i] ? run.engagementHigh[d] : run.engagementLow[d]) += prob / (high[i] ? nHigh : nLow);
      if (rngs[i].bernoulli(prob)) {
        if (rewards) {
          points[i] += 1.0;
          habit[i] += p.habitRate * (p.habitMax - habit[i]);
          if (intrinsic[i] > 0.0) intrinsic[i] = std::max(0.0, intrinsic[i] - p.undermining * (1.0 + intrinsic[i]));
        } else {
          habit[i] -= p.habitDecay * habit[i];
        }
      }
      run.points[d] += points[i] / p.students;
      run.intrinsic[d] += intrinsic[i] / p.students;
    }
    const int phase = d < p.rewardStart ? 0 : (rewards ? 1 : 2);
    sums[phase] += run.engagement[d];
    ++counts[phase];
  }
  run.before = counts[0] ? sums[0] / counts[0] : 0.0;
  run.during = counts[1] ? sums[1] / counts[1] : 0.0;
  run.after = counts[2] ? sums[2] / counts[2] : 0.0;
  return run;
}

}  // namespace syn
