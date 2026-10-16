#include "syn/social/dialogue.hpp"

#include <algorithm>
#include <numeric>
#include <stdexcept>

#include "syn/core/math.hpp"
#include "syn/core/random.hpp"

namespace syn {

const char* groupingName(Grouping g) {
  switch (g) {
    case Grouping::None: return "individual work (no dialogue)";
    case Grouping::Random: return "random groups";
    case Grouping::MixedAbility: return "mixed-ability groups";
    case Grouping::SameAbility: return "same-ability groups";
  }
  return "?";
}

namespace {

// Partitions learners into groups of about `size` according to the grouping rule.
std::vector<std::vector<int>> formGroups(const std::vector<double>& u, int size, Grouping g, Rng& rng) {
  const int n = static_cast<int>(u.size());
  std::vector<int> order(n);
  std::iota(order.begin(), order.end(), 0);
  // Always draw the shuffle so every grouping consumes the same random numbers.
  for (int i = n - 1; i > 0; --i) std::swap(order[i], order[rng.uniformInt(i + 1)]);
  if (g == Grouping::MixedAbility || g == Grouping::SameAbility)
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return u[a] > u[b]; });
  const int groups = std::max(1, (n + size - 1) / size);
  std::vector<std::vector<int>> out(groups);
  for (int k = 0; k < n; ++k) {
    // Mixed ability deals ranked learners round-robin (each group spans the range);
    // same ability takes consecutive ranks.
    const int gi = g == Grouping::SameAbility ? std::min(groups - 1, k / size) : k % groups;
    out[gi].push_back(order[k]);
  }
  return out;
}

}  // namespace

DialogueRun simulateDialogue(const DialogueParams& p, std::uint64_t seed) {
  if (p.learners < 2 || p.groupSize < 2 || p.sessions < 1 || p.classes < 1) throw std::invalid_argument("simulateDialogue: bad sizes");
  DialogueRun run;
  run.grouping = groupingName(p.grouping);
  const auto steps = static_cast<std::size_t>(p.sessions) + 1;
  run.meanUnderstanding.assign(steps, 0.0);
  run.highCapital.assign(steps, 0.0);
  run.lowCapital.assign(steps, 0.0);
  run.spread.assign(steps, 0.0);
  const double inclusion = clamp01(p.inclusivity);
  for (int cls = 0; cls < p.classes; ++cls) {
    Rng rng = Rng::stream(seed, static_cast<std::uint64_t>(cls));
    std::vector<double> c(p.learners), u(p.learners);
    for (int i = 0; i < p.learners; ++i) {
      c[i] = clamp01(rng.normal(p.capitalMean, p.capitalSd));
      u[i] = clamp01(p.priorBase + p.priorCapital * c[i] + 0.08 * rng.normal());
    }
    const double median = quantile(c, 0.5);
    std::vector<bool> high(p.learners);
    int nHigh = 0;
    for (int i = 0; i < p.learners; ++i) nHigh += (high[i] = c[i] > median) ? 1 : 0;
    const int nLow = p.learners - nHigh;
    auto record = [&](std::size_t s) {
      double hi = 0.0, lo = 0.0;
      for (int i = 0; i < p.learners; ++i) (high[i] ? hi : lo) += u[i];
      run.meanUnderstanding[s] += mean(u) / p.classes;
      run.highCapital[s] += (nHigh ? hi / nHigh : 0.0) / p.classes;
      run.lowCapital[s] += (nLow ? lo / nLow : 0.0) / p.classes;
      run.spread[s] += stdev(u) / p.classes;
    };
    record(0);
    for (int s = 1; s <= p.sessions; ++s) {
      std::vector<double> next = u;
      for (int i = 0; i < p.learners; ++i) {
        const double access = 1.0 - p.capitalPenalty * (1.0 - c[i]) * (1.0 - inclusion);
        next[i] += p.instructionRate * std::max(0.0, access) * (1.0 - u[i]) * (1.0 + 0.3 * rng.normal());
      }
      const auto groups = formGroups(u, p.groupSize, p.grouping, rng);
      if (p.grouping != Grouping::None) {
        for (const auto& g : groups) {
          const double m = static_cast<double>(g.size()) - 1.0;
          if (m <= 0) continue;
          for (int i : g) {
            double learnFrom = 0.0, explainTo = 0.0;
            for (int j : g) {
              if (i == j) continue;
              learnFrom += std::max(0.0, u[j] - u[i]);
              explainTo += std::max(0.0, u[i] - u[j]);
            }
            next[i] += p.peerRate * learnFrom / m + p.articulationRate * explainTo / m * (1.0 - u[i]);
          }
        }
      }
      for (int i = 0; i < p.learners; ++i) u[i] = clamp01(next[i]);
      record(static_cast<std::size_t>(s));
    }
  }
  run.finalMean = run.meanUnderstanding.back();
  run.finalGap = run.highCapital.back() - run.lowCapital.back();
  return run;
}

}  // namespace syn
