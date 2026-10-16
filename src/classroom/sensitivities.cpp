#include "syn/classroom/sensitivities.hpp"

#include <cmath>
#include <stdexcept>

#include "syn/core/math.hpp"
#include "syn/core/random.hpp"

namespace syn {

namespace {
// Run layout: [0] the design; then per lever k and replication r: up (CRN seed), down (CRN
// seed), down (independent seed).
std::size_t runIndex(int k, int rep, int which, int reps) { return 1 + (static_cast<std::size_t>(k) * reps + rep) * 3 + which; }
}  // namespace

SensitivityRequest sensitivityRequest(const InstructionalDesign& design, double bump, int replications, std::uint64_t baseSeed) {
  if (!(bump > 0.0 && bump <= 0.5) || replications < 2) throw std::invalid_argument("leverSensitivities: need 0 < bump <= 0.5 and >= 2 replications");
  SensitivityRequest q;
  q.design = design.clamped();
  q.bump = bump;
  q.replications = replications;
  q.baseSeed = baseSeed;
  q.runs.push_back({q.design, baseSeed});
  for (int k = 0; k < kLevers; ++k) {
    const double v = q.design.lever[k];
    InstructionalDesign up = q.design, down = q.design;
    up.lever[k] = std::min(1.0, v + bump);
    down.lever[k] = std::max(0.0, v - bump);
    for (int rep = 0; rep < replications; ++rep) {
      const std::uint64_t seed = streamSeed(baseSeed, 1000 + static_cast<std::uint64_t>(rep));
      const std::uint64_t other = streamSeed(baseSeed, 5000 + static_cast<std::uint64_t>(rep));
      q.runs.push_back({up, seed});
      q.runs.push_back({down, seed});
      q.runs.push_back({down, other});
    }
  }
  return q;
}

SensitivityResult sensitivitiesFromScores(const SensitivityRequest& q, const std::vector<double>& scores) {
  if (scores.size() != q.runs.size()) throw std::invalid_argument("sensitivitiesFromScores: one score per run needed");
  SensitivityResult r;
  r.bump = q.bump;
  r.replications = q.replications;
  r.score = scores[0];
  double ratioSum = 0.0;
  int ratioCount = 0;
  for (int k = 0; k < kLevers; ++k) {
    const double v = q.design.lever[k];
    const double width = std::min(1.0, v + q.bump) - std::max(0.0, v - q.bump);
    std::vector<double> crn, ind;
    for (int rep = 0; rep < q.replications; ++rep) {
      const double up = scores[runIndex(k, rep, 0, q.replications)];
      crn.push_back((up - scores[runIndex(k, rep, 1, q.replications)]) / width);
      ind.push_back((up - scores[runIndex(k, rep, 2, q.replications)]) / width);
    }
    LeverSensitivity s;
    s.lever = static_cast<Lever>(k);
    s.value = v;
    s.crn = mean(crn);
    s.crnStdError = stdev(crn) / std::sqrt(static_cast<double>(q.replications));
    s.independent = mean(ind);
    s.independentStdError = stdev(ind) / std::sqrt(static_cast<double>(q.replications));
    if (s.crnStdError > 0.0) {
      ratioSum += (s.independentStdError / s.crnStdError) * (s.independentStdError / s.crnStdError);
      ++ratioCount;
    }
    r.levers.push_back(s);
  }
  r.meanVarianceReduction = ratioCount ? ratioSum / ratioCount : 0.0;
  return r;
}

SensitivityResult leverSensitivities(const ClassroomEngine& engine, const InstructionalDesign& design, double bump,
                                     int replications) {
  const SensitivityRequest q = sensitivityRequest(design, bump, replications, engine.settings().seed);
  std::vector<double> scores;
  for (const auto& run : q.runs) scores.push_back(engine.objective(run.design, run.seed));
  return sensitivitiesFromScores(q, scores);
}

}  // namespace syn
