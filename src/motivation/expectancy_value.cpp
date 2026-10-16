#include "syn/motivation/expectancy_value.hpp"

#include <cmath>
#include <stdexcept>

#include "syn/core/math.hpp"
#include "syn/core/random.hpp"

namespace syn {

double& MotivationLevers::operator[](int k) {
  switch (k) {
    case 0: return relevance;
    case 1: return challengeFit;
    case 2: return choice;
    case 3: return successSetup;
    case 4: return selfAssessment;
  }
  throw std::out_of_range("MotivationLevers index");
}

double MotivationLevers::operator[](int k) const { return const_cast<MotivationLevers&>(*this)[k]; }

const char* MotivationLevers::name(int k) {
  static const char* names[kCount] = {"emphasise relevance", "appropriate challenge", "choice and control",
                                      "set up for success", "guided self-assessment"};
  return k >= 0 && k < kCount ? names[k] : "?";
}

double challengeInterest(double offset) { return 0.2 - 0.12 * (offset - 0.6) * (offset - 0.6); }

TermRun simulateTerm(const MotivationLevers& lv, const ExpectancyValueParams& p, std::uint64_t seed) {
  if (p.learners < 1 || p.weeks < 1) throw std::invalid_argument("simulateTerm: bad sizes");
  TermRun run;
  const auto w = static_cast<std::size_t>(p.weeks);
  for (auto* v : {&run.motivation, &run.value, &run.efficacy, &run.ability, &run.calibrationError, &run.engaged}) v->assign(w, 0.0);
  const double relevance = clamp01(lv.relevance), fit = clamp01(lv.challengeFit), choice = clamp01(lv.choice),
               setup = clamp01(lv.successSetup), assess = clamp01(lv.selfAssessment);
  const double n = p.learners;
  for (int i = 0; i < p.learners; ++i) {
    Rng rng = Rng::stream(seed, static_cast<std::uint64_t>(i));
    double ability = rng.normal(p.abilityMean, p.abilitySd);
    const double start = ability;
    double efficacy = clamp01(rng.normal(p.efficacyMean, p.efficacySd) + 0.1 * choice);
    double bias = p.biasSd * rng.normal();
    bool engaged = true;
    int weakWeeks = 0;
    double m = 0.0;
    for (std::size_t t = 0; t < w; ++t) {
      // A poorly matched task misses the stretch zone by a random amount (too easy or too hard).
      const double miss = rng.normal();
      const double offset = 0.6 + (1.0 - fit) * 1.6 * miss;
      const double value = clamp(p.baseValue + 0.3 * relevance + 0.12 * choice + challengeInterest(offset), 0.0, 1.0);
      m = engaged ? value * efficacy : 0.0;
      weakWeeks = m < p.dropoutBelow ? weakWeeks + 1 : 0;
      if (weakWeeks >= 3) engaged = false;
      const double effort = engaged ? m : 0.0;
      const double pSuccess = logistic(1.6 * effort + 0.5 * setup - offset);
      const double outcome = rng.uniform() < pSuccess ? 1.0 : 0.0;
      if (engaged) {
        ability += 0.25 * effort * (1.0 + 0.3 * setup);
        // Self-efficacy follows experience, read through the learner's perception bias.
        const double perceived = clamp01(outcome + bias * 0.5);
        efficacy = clamp01(efficacy + (0.12 + 0.12 * setup) * (perceived - efficacy));
        bias *= 1.0 - 0.15 * assess;
      }
      run.motivation[t] += m / n;
      run.value[t] += value / n;
      run.efficacy[t] += efficacy / n;
      run.ability[t] += (ability - start) / n;
      run.calibrationError[t] += std::fabs(bias) / n;
      run.engaged[t] += (engaged ? 1.0 : 0.0) / n;
    }
    run.finalAbilityGain += (ability - start) / n;
  }
  run.finalMotivation = run.motivation.back();
  run.finalEngaged = run.engaged.back();
  return run;
}

}  // namespace syn
