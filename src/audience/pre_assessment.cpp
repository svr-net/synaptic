#include "syn/audience/pre_assessment.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "syn/core/math.hpp"
#include "syn/core/random.hpp"
#include "syn/social/zpd.hpp"

namespace syn {

PriorSource PriorSource::assumption() { return {"assumption (no information)", 0.8, 1.5}; }
PriorSource PriorSource::research() { return {"research literature", 0.25, 0.7}; }
PriorSource PriorSource::demographics() { return {"demographics and campus offices", 0.1, 0.6}; }
PriorSource PriorSource::curriculum() { return {"curriculum / course instructor", 0.0, 0.35}; }
std::vector<PriorSource> PriorSource::all() { return {assumption(), research(), demographics(), curriculum()}; }

//                                          name                class  read/learner noise recall social particip. indiv. affect engage interests anxiety
PreAssessment PreAssessment::none() { return {"no pre-assessment", 0, 0, 1.0, 0, 0, 0, false, 0, 0, 0, 0}; }
PreAssessment PreAssessment::showOfHands() { return {"show of hands", 2, 0, 1.2, 0.4, 0.3, 0.9, false, 0.2, 0.2, 0.0, 0.0}; }
PreAssessment PreAssessment::anonymousPoll() { return {"anonymous poll", 3, 0, 1.0, 0.4, 0.0, 0.8, false, 0.4, 0.3, 0.1, 0.0}; }
PreAssessment PreAssessment::worksheet() { return {"worksheet", 8, 0.25, 0.6, 0.0, 0.0, 0.95, true, 0.1, 0.3, 0.1, 0.0}; }
PreAssessment PreAssessment::kwl() { return {"K-W-L chart", 7, 0.5, 0.9, 0.2, 0.1, 0.9, true, 0.6, 0.6, 0.7, 0.0}; }
PreAssessment PreAssessment::conceptMap() { return {"concept map", 8, 0.6, 0.7, 0.0, 0.1, 0.9, true, 0.2, 0.4, 0.4, 0.0}; }
PreAssessment PreAssessment::brainstorm() { return {"brainstorm", 5, 0.15, 1.1, 0.3, 0.2, 0.85, false, 0.3, 0.5, 0.5, 0.0}; }
PreAssessment PreAssessment::quiz() { return {"formal quiz", 12, 0.1, 0.5, 0.3, 0.0, 0.95, true, 0.0, 0.0, 0.0, 0.3}; }
PreAssessment PreAssessment::homework() { return {"pre-assessment as homework", 0, 0, 0.6, 0.0, 0.0, 0.3, true, 0.3, 0.3, 0.3, 0.0}; }
std::vector<PreAssessment> PreAssessment::all() {
  return {none(), showOfHands(), anonymousPoll(), worksheet(), kwl(), conceptMap(), brainstorm(), quiz(), homework()};
}

OneShotResult simulateOneShot(const OneShotParams& p, const PriorSource& prior, const PreAssessment& inst, std::uint64_t seed) {
  if (p.classes < 1 || p.classSize < 1 || p.groups < 1 || p.sessionMinutes <= 0.0 || prior.sd <= 0.0)
    throw std::invalid_argument("simulateOneShot: bad parameters");
  OneShotResult r;
  r.prior = prior.name;
  r.instrument = inst.name;
  ZpdParams zpd;
  zpd.scaffold = p.scaffold;
  if (p.sharpness <= 0.0) throw std::invalid_argument("simulateOneShot: sharpness must be positive");
  const double sharp = p.sharpness;
  const double offset = p.scaffold / (2.0 * sharp);  // the ZPD gain peaks at k(b − θ) = s/2
  const double sigma2 = p.withinClassSd * p.withinClassSd, noise2 = inst.noise * inst.noise;
  const double reliability = sigma2 / (sigma2 + noise2);
  const double m0 = p.communityMean + prior.bias, s02 = prior.sd * prior.sd;
  const std::size_t n = static_cast<std::size_t>(p.classSize);
  double learners = 0.0, sqErr = 0.0;

  for (int c = 0; c < p.classes; ++c) {
    Rng rng = Rng::stream(seed, static_cast<std::uint64_t>(c));
    const double mu = rng.normal(p.communityMean, p.betweenClassSd);
    std::vector<double> theta(n), response(n);
    std::vector<bool> responds(n);
    for (std::size_t i = 0; i < n; ++i) {  // fixed draws per learner: common random numbers across instruments
      theta[i] = rng.normal(mu, p.withinClassSd);
      responds[i] = rng.uniform() < inst.participation;
      response[i] = theta[i] + inst.recallBias + inst.socialBias + inst.noise * rng.normal();
    }
    // Responses read within the analysis budget.
    std::vector<std::size_t> read;
    for (std::size_t i = 0; i < n; ++i)
      if (responds[i]) read.push_back(i);
    double analysis = 0.0;
    if (inst.analysisPerLearner > 0.0 && !read.empty()) {
      const auto canRead = static_cast<std::size_t>(std::floor(p.analysisBudget / inst.analysisPerLearner));
      if (read.size() > canRead) read.resize(canRead);
      analysis = inst.analysisPerLearner * static_cast<double>(read.size());
    }
    r.analysedShare += static_cast<double>(read.size()) / static_cast<double>(n) / p.classes;

    // The instructor's estimate of the class: prior and measurement weighted by precision.
    double estimate = m0, mean = 0.0;
    if (!read.empty()) {
      for (auto i : read) mean += response[i];
      mean /= static_cast<double>(read.size());
      const double v = (sigma2 + noise2) / static_cast<double>(read.size());
      estimate = (m0 / s02 + mean / v) / (1.0 / s02 + 1.0 / v);
    }
    sqErr += (estimate - mu) * (estimate - mu);
    r.pitchError += std::fabs(estimate - mu) / p.classes;

    // Pitch: one level for the class, or one per group of pre-assessment scores.
    std::vector<double> level(n, estimate + offset);
    int groups = 1;
    if (inst.individual && p.groups > 1 && read.size() >= static_cast<std::size_t>(p.groups)) {
      groups = p.groups;
      std::vector<std::size_t> order = read;
      std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) { return response[a] < response[b]; });
      for (int g = 0; g < groups; ++g) {
        const std::size_t lo = order.size() * g / groups, hi = order.size() * (g + 1) / groups;
        double gm = 0.0;
        for (std::size_t k = lo; k < hi; ++k) gm += response[order[k]];
        gm /= static_cast<double>(hi - lo);
        for (std::size_t k = lo; k < hi; ++k) level[order[k]] = estimate + reliability * (gm - mean) + offset;
      }
    }
    const double efficiency = 1.0 / (1.0 + p.groupCost * (groups - 1));
    const double teaching = std::max(0.0, p.sessionMinutes - inst.classMinutes - 0.5 * analysis);
    const double anxiety = clamp01(p.anxietyMean * (1.0 - 0.5 * inst.affective) + inst.anxiety);
    const double boost = (1.0 + 0.25 * inst.engagement) * (1.0 + 0.2 * inst.interests) * (1.0 - 0.3 * anxiety);
    const double plainBoost = 1.0 - 0.3 * clamp01(p.anxietyMean);
    for (std::size_t i = 0; i < n; ++i) {
      const ZpdAssessment task = assessTask(sharp * theta[i], sharp * level[i], zpd);
      const double gain = teaching / p.sessionMinutes * efficiency * zpd.learningRate * task.assisted * (1.0 - task.alone) * boost;
      const ZpdAssessment plain = assessTask(sharp * theta[i], sharp * (m0 + offset), zpd);  // no pre-assessment: pitched at the prior
      const double plainGain = zpd.learningRate * plain.assisted * (1.0 - plain.alone) * plainBoost;
      r.meanGain += gain;
      r.gainWithoutPreAssessment += plainGain;
      r.zpdShare += task.zone == Zone::Proximal ? 1.0 : 0.0;
      learners += 1.0;
    }
    r.teachingMinutes += teaching / p.classes;
  }
  r.meanGain /= learners;
  r.gainWithoutPreAssessment /= learners;
  r.zpdShare /= learners;
  r.estimateRmse = std::sqrt(sqErr / p.classes);
  r.valueOfInformation = r.meanGain - r.gainWithoutPreAssessment;
  return r;
}

}  // namespace syn
