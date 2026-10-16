#include "syn/classroom/allocation.hpp"

#include <cmath>
#include <map>
#include <numeric>
#include <stdexcept>

#include "syn/core/random.hpp"

namespace syn {

AllocationRequest allocationRequest(const InstructionalDesign& design, const InstructionalDesign& baseline, int permutations,
                                    std::uint64_t orderSeed) {
  if (permutations < 1) throw std::invalid_argument("allocateLearningGain: need at least one permutation");
  AllocationRequest q;
  q.design = design.clamped();
  q.baseline = baseline.clamped();
  q.permutations = permutations;
  for (int k = 0; k < kLevers; ++k)
    if (std::fabs(q.design.lever[k] - q.baseline.lever[k]) > 1e-12) q.active.push_back(k);
  if (q.active.size() > 60) throw std::invalid_argument("allocateLearningGain: too many levers");
  const std::size_t m = q.active.size();
  const unsigned long long full = m == 0 ? 0ULL : (~0ULL >> (64 - m));
  std::map<unsigned long long, bool> seen;
  auto need = [&](unsigned long long mask) {
    if (seen.emplace(mask, true).second) q.masks.push_back(mask);
  };
  need(0ULL);
  need(full);
  Rng rng(orderSeed);
  for (int p = 0; p < permutations; ++p) {
    std::vector<int> order(m);
    std::iota(order.begin(), order.end(), 0);
    for (int i = static_cast<int>(m) - 1; i > 0; --i) std::swap(order[i], order[rng.uniformInt(i + 1)]);
    unsigned long long mask = 0ULL;
    for (int j : order) need(mask |= 1ULL << j);
    q.orders.push_back(order);
  }
  for (std::size_t j = 0; j < m; ++j) {
    need(full & ~(1ULL << j));  // incremental
    need(1ULL << j);            // standalone
  }
  for (auto mask : q.masks) {
    InstructionalDesign d = q.baseline;
    for (std::size_t j = 0; j < m; ++j)
      if (mask >> j & 1ULL) d.lever[q.active[j]] = q.design.lever[q.active[j]];
    q.designs.push_back(d);
  }
  return q;
}

AllocationResult allocateFromScores(const AllocationRequest& q, const std::vector<double>& scores) {
  if (scores.size() != q.masks.size()) throw std::invalid_argument("allocateFromScores: one score per design needed");
  std::map<unsigned long long, double> f;
  for (std::size_t i = 0; i < q.masks.size(); ++i) f[q.masks[i]] = scores[i];
  const std::size_t m = q.active.size();
  const unsigned long long full = m == 0 ? 0ULL : (~0ULL >> (64 - m));
  AllocationResult r;
  r.permutations = q.permutations;
  r.evaluations = static_cast<int>(q.masks.size());
  r.baselineScore = f.at(0ULL);
  r.designScore = f.at(full);
  r.totalGain = r.designScore - r.baselineScore;
  std::vector<double> shapley(m, 0.0);
  for (const auto& order : q.orders) {
    unsigned long long mask = 0ULL;
    double prev = r.baselineScore;
    for (int j : order) {
      mask |= 1ULL << j;
      const double v = f.at(mask);
      shapley[j] += (v - prev) / q.permutations;
      prev = v;
    }
  }
  for (std::size_t j = 0; j < m; ++j) {
    LeverContribution c;
    c.lever = static_cast<Lever>(q.active[j]);
    c.baselineValue = q.baseline.lever[q.active[j]];
    c.designValue = q.design.lever[q.active[j]];
    c.shapley = shapley[j];
    c.incremental = r.designScore - f.at(full & ~(1ULL << j));
    c.standalone = f.at(1ULL << j) - r.baselineScore;
    r.shapleySum += c.shapley;
    r.levers.push_back(c);
  }
  return r;
}

AllocationResult allocateLearningGain(const ClassroomEngine& engine, const InstructionalDesign& design,
                                      const InstructionalDesign& baseline, int permutations, std::uint64_t orderSeed) {
  const AllocationRequest q = allocationRequest(design, baseline, permutations, orderSeed);
  std::vector<double> scores;
  for (const auto& d : q.designs) scores.push_back(engine.objective(d));
  return allocateFromScores(q, scores);
}

}  // namespace syn
