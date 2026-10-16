#include "syn/classroom/sensitivities.hpp"

#include <cmath>
#include <stdexcept>

#include "syn/core/math.hpp"
#include "syn/core/random.hpp"

namespace syn {

SensitivityResult leverSensitivities(const ClassroomEngine& engine, const InstructionalDesign& design, double bump,
                                     int replications) {
  if (!(bump > 0.0 && bump <= 0.5) || replications < 2) throw std::invalid_argument("leverSensitivities: need 0 < bump <= 0.5 and >= 2 replications");
  const InstructionalDesign x = design.clamped();
  SensitivityResult r;
  r.bump = bump;
  r.replications = replications;
  r.score = engine.objective(x);
  const std::uint64_t base = engine.settings().seed;
  double ratioSum = 0.0;
  int ratioCount = 0;
  for (int k = 0; k < kLevers; ++k) {
    const double v = x.lever[k];
    const double up = std::min(1.0, v + bump), down = std::max(0.0, v - bump);
    InstructionalDesign dUp = x, dDown = x;
    dUp.lever[k] = up;
    dDown.lever[k] = down;
    std::vector<double> crn, ind;
    for (int rep = 0; rep < replications; ++rep) {
      const std::uint64_t seed = streamSeed(base, 1000 + static_cast<std::uint64_t>(rep));
      crn.push_back((engine.objective(dUp, seed) - engine.objective(dDown, seed)) / (up - down));
      const std::uint64_t other = streamSeed(base, 5000 + static_cast<std::uint64_t>(rep));
      ind.push_back((engine.objective(dUp, seed) - engine.objective(dDown, other)) / (up - down));
    }
    LeverSensitivity s;
    s.lever = static_cast<Lever>(k);
    s.value = v;
    s.crn = mean(crn);
    s.crnStdError = stdev(crn) / std::sqrt(static_cast<double>(replications));
    s.independent = mean(ind);
    s.independentStdError = stdev(ind) / std::sqrt(static_cast<double>(replications));
    if (s.crnStdError > 0.0) {
      ratioSum += (s.independentStdError / s.crnStdError) * (s.independentStdError / s.crnStdError);
      ++ratioCount;
    }
    r.levers.push_back(s);
  }
  r.meanVarianceReduction = ratioCount ? ratioSum / ratioCount : 0.0;
  return r;
}

}  // namespace syn
