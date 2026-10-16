#include "syn/motivation/mindset.hpp"

#include <algorithm>
#include <stdexcept>

#include "syn/core/math.hpp"
#include "syn/core/random.hpp"

namespace syn {

const char* feedbackStyleName(FeedbackStyle f) {
  switch (f) {
    case FeedbackStyle::Ability: return "praise ability";
    case FeedbackStyle::Neutral: return "grades only";
    case FeedbackStyle::ProcessNotYet: return "praise process, \"not yet\"";
  }
  return "?";
}

MindsetRun simulateMindset(const MindsetParams& p, std::uint64_t seed) {
  if (p.learners < 1 || p.weeks < 1 || p.strategies < 1) throw std::invalid_argument("simulateMindset: bad sizes");
  MindsetRun run;
  const auto w = static_cast<std::size_t>(p.weeks);
  for (auto* v : {&run.skillFixed, &run.skillGrowth, &run.persisting, &run.mindset, &run.challengeSeeking}) v->assign(w, 0.0);
  int nFixed = 0, nGrowth = 0;
  std::vector<double> fixedSum(w, 0.0), growthSum(w, 0.0);
  const double n = p.learners;
  for (int i = 0; i < p.learners; ++i) {
    Rng rng = Rng::stream(seed, static_cast<std::uint64_t>(i));
    double g = clamp01(rng.normal(p.growthMean, p.growthSd));
    const bool startsGrowth = g >= 0.5;
    (startsGrowth ? nGrowth : nFixed) += 1;
    std::vector<double> quality(p.strategies);
    for (auto& q : quality) q = rng.uniform(0.2, 1.0);
    int strategy = 0;
    double skill = 0.0, effort = 0.6;
    bool engaged = true;
    for (std::size_t t = 0; t < w; ++t) {
      const double uChoice = rng.uniform(), uOutcome = rng.uniform(), uSwitch = rng.uniform(), uQuit = rng.uniform();
      const int pick = rng.uniformInt(p.strategies);
      const bool challenge = uChoice < 0.15 + 0.7 * g;
      if (engaged) {
        const double difficulty = (challenge ? 1.0 : -0.5) - (p.highStandards ? 0.0 : 0.8);
        const double success = logistic(1.5 * effort - difficulty);
        const bool ok = uOutcome < success;
        const double stretch = std::max(0.2, (challenge ? 1.6 : 0.6) - (p.highStandards ? 0.0 : 0.3));
        skill += p.learningRate * effort * quality[strategy] * stretch;
        if (ok) {
          if (p.feedback == FeedbackStyle::Ability) g -= 0.03 * g;
          if (p.feedback == FeedbackStyle::ProcessNotYet) g += 0.02 * (1.0 - g);
        } else {
          if (p.feedback == FeedbackStyle::ProcessNotYet) g += 0.04 * (1.0 - g);
          // Growth: more effort, a different strategy, feedback used. Fixed: withdraw effort.
          effort = clamp(effort + 0.12 * g - 0.18 * (1.0 - g), 0.0, 1.0);
          if (uSwitch < 0.6 * g && quality[pick] >= quality[strategy] - 0.1) strategy = pick;
          if (effort < 0.2 && uQuit < 0.5 * (1.0 - g)) engaged = false;
        }
        g = clamp01(g + 0.04 * clamp01(p.coaching) * (1.0 - g));
      }
      run.persisting[t] += (engaged ? 1.0 : 0.0) / n;
      run.mindset[t] += g / n;
      run.challengeSeeking[t] += (engaged && challenge ? 1.0 : 0.0) / n;
      (startsGrowth ? growthSum : fixedSum)[t] += skill;
    }
    run.finalSkillGain += skill / n;
  }
  for (std::size_t t = 0; t < w; ++t) {
    run.skillFixed[t] = nFixed ? fixedSum[t] / nFixed : 0.0;
    run.skillGrowth[t] = nGrowth ? growthSum[t] / nGrowth : 0.0;
  }
  run.finalPersisting = run.persisting.back();
  return run;
}

}  // namespace syn
