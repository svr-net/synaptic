#include "syn/classroom/allocation.hpp"

#include <cmath>
#include <map>
#include <numeric>
#include <stdexcept>

#include "syn/core/random.hpp"

namespace syn {

AllocationResult allocateLearningGain(const ClassroomEngine& engine, const InstructionalDesign& design,
                                      const InstructionalDesign& baseline, int permutations, std::uint64_t orderSeed) {
  if (permutations < 1) throw std::invalid_argument("allocateLearningGain: need at least one permutation");
  const InstructionalDesign x = design.clamped(), b = baseline.clamped();
  std::vector<int> active;
  for (int k = 0; k < kLevers; ++k)
    if (std::fabs(x.lever[k] - b.lever[k]) > 1e-12) active.push_back(k);

  AllocationResult r;
  // Memoise evaluations by the set of switched levers (a bit mask over `active`).
  std::map<unsigned long long, double> cache;
  auto f = [&](unsigned long long mask) {
    auto it = cache.find(mask);
    if (it != cache.end()) return it->second;
    InstructionalDesign d = b;
    for (std::size_t j = 0; j < active.size(); ++j)
      if (mask >> j & 1ULL) d.lever[active[j]] = x.lever[active[j]];
    const double v = engine.objective(d);
    ++r.evaluations;
    cache.emplace(mask, v);
    return v;
  };
  if (active.size() > 60) throw std::invalid_argument("allocateLearningGain: too many levers");
  const unsigned long long full = active.empty() ? 0ULL : (~0ULL >> (64 - active.size()));
  r.baselineScore = f(0ULL);
  r.designScore = f(full);
  r.totalGain = r.designScore - r.baselineScore;

  std::vector<double> shapley(active.size(), 0.0);
  Rng rng(orderSeed);
  std::vector<int> order(active.size());
  for (int p = 0; p < permutations; ++p) {
    std::iota(order.begin(), order.end(), 0);
    for (int i = static_cast<int>(order.size()) - 1; i > 0; --i) std::swap(order[i], order[rng.uniformInt(i + 1)]);
    unsigned long long mask = 0ULL;
    double prev = r.baselineScore;
    for (int j : order) {
      mask |= 1ULL << j;
      const double v = f(mask);
      shapley[j] += (v - prev) / permutations;
      prev = v;
    }
  }
  r.permutations = permutations;
  for (std::size_t j = 0; j < active.size(); ++j) {
    LeverContribution c;
    c.lever = static_cast<Lever>(active[j]);
    c.baselineValue = b.lever[active[j]];
    c.designValue = x.lever[active[j]];
    c.shapley = shapley[j];
    c.incremental = r.designScore - f(full & ~(1ULL << j));
    c.standalone = f(1ULL << j) - r.baselineScore;
    r.shapleySum += c.shapley;
    r.levers.push_back(c);
  }
  return r;
}

}  // namespace syn
