#include "syn/audience/community.hpp"

#include <map>
#include <stdexcept>

#include "syn/core/math.hpp"
#include "syn/core/random.hpp"

namespace syn {

std::vector<CommunityAttribute> defaultCommunityAttributes() {
  return {
      {"nontraditional", 0.7, -0.15},  // works full time, caregiver, delayed entry...
      {"first-generation", 0.3, -0.35},
      {"English learner", 0.15, -0.4},
      {"older adult", 0.2, -0.25},
      {"disability", 0.12, 0.0},
  };
}

namespace {
double r2(const std::vector<double>& y, const std::vector<double>& prediction) {
  const double m = mean(y);
  double sse = 0.0, sst = 0.0;
  for (std::size_t i = 0; i < y.size(); ++i) {
    sse += (y[i] - prediction[i]) * (y[i] - prediction[i]);
    sst += (y[i] - m) * (y[i] - m);
  }
  return sst > 0.0 ? 1.0 - sse / sst : 0.0;
}
}  // namespace

CommunityResult simulateCommunity(const CommunityParams& p, std::uint64_t seed) {
  const std::size_t k = p.attributes.size();
  if (p.learners < 2 || k == 0 || k > 20) throw std::invalid_argument("simulateCommunity: bad sizes");
  const auto n = static_cast<std::size_t>(p.learners);
  std::vector<unsigned> cell(n);
  std::vector<double> theta(n), measured(n);
  for (std::size_t i = 0; i < n; ++i) {
    Rng rng = Rng::stream(seed, i);
    unsigned mask = 0;
    double t = 0.0;
    for (std::size_t a = 0; a < k; ++a)
      if (rng.uniform() < p.attributes[a].prevalence) {
        mask |= 1u << a;
        t += p.attributes[a].effect;
      }
    cell[i] = mask;
    theta[i] = t + p.individualSd * rng.normal();
    measured[i] = theta[i] + p.measurementNoise * rng.normal();
  }
  CommunityResult r;
  // One attribute at a time: predict each learner by the mean of their group (has / has not).
  for (std::size_t a = 0; a < k; ++a) {
    double s1 = 0, n1 = 0, s0 = 0, n0 = 0;
    for (std::size_t i = 0; i < n; ++i)
      if (cell[i] >> a & 1u) {
        s1 += theta[i];
        ++n1;
      } else {
        s0 += theta[i];
        ++n0;
      }
    std::vector<double> pred(n);
    for (std::size_t i = 0; i < n; ++i) pred[i] = (cell[i] >> a & 1u) ? (n1 ? s1 / n1 : 0.0) : (n0 ? s0 / n0 : 0.0);
    r.attributes.push_back(p.attributes[a].name);
    r.groupShare.push_back(n1 / static_cast<double>(n));
    r.groupMean.push_back(n1 ? s1 / n1 : 0.0);
    r.singleR2.push_back(r2(theta, pred));
    r.bestSingleR2 = std::max(r.bestSingleR2, r.singleR2.back());
  }
  // Every intersection of attributes.
  std::map<unsigned, std::pair<double, int>> cells;
  for (std::size_t i = 0; i < n; ++i) {
    cells[cell[i]].first += theta[i];
    ++cells[cell[i]].second;
  }
  r.cells = static_cast<int>(cells.size());
  std::vector<double> cellPred(n), pre(n), combined(n);
  const double s2 = p.individualSd * p.individualSd, e2 = p.measurementNoise * p.measurementNoise;
  for (std::size_t i = 0; i < n; ++i) {
    const auto& c = cells[cell[i]];
    cellPred[i] = c.first / c.second;
    // The pre-assessment alone: the response shrunk towards the community mean by its reliability.
    pre[i] = s2 / (s2 + e2) * measured[i];
    // Both: the cell mean as the prior for the individual, the response as the measurement.
    combined[i] = (cellPred[i] / s2 + measured[i] / e2) / (1.0 / s2 + 1.0 / e2);
  }
  r.intersectionR2 = r2(theta, cellPred);
  r.preAssessmentR2 = r2(theta, pre);
  r.combinedR2 = r2(theta, combined);
  // Privacy: suppress cells smaller than k.
  for (int minSize : {1, 5, 10, 20, 50}) {
    int small = 0, covered = 0;
    for (const auto& [mask, c] : cells)
      if (c.second < minSize) {
        ++small;
        covered += c.second;
      }
    r.minCellSizes.push_back(minSize);
    r.suppressedCells.push_back(static_cast<double>(small) / static_cast<double>(cells.size()));
    r.suppressedLearners.push_back(static_cast<double>(covered) / static_cast<double>(n));
  }
  return r;
}

}  // namespace syn
