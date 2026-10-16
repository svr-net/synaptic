#include "syn/behaviorism/classical_conditioning.hpp"

#include <cmath>
#include <tuple>
#include <stdexcept>

#include "syn/core/elliptic_cga.hpp"

namespace syn {

RescorlaWagner::RescorlaWagner(std::vector<ConditionedStimulus> stimuli, double betaUs, double betaNoUs, double lambda)
    : stimuli_(std::move(stimuli)), v_(stimuli_.size(), 0.0), betaUs_(betaUs), betaNoUs_(betaNoUs), lambda_(lambda) {
  for (const auto& s : stimuli_)
    if (!(s.salience > 0.0 && s.salience <= 1.0)) throw std::invalid_argument("RescorlaWagner: salience must lie in (0, 1]");
  if (!(betaUs > 0.0 && betaUs <= 1.0) || !(betaNoUs > 0.0 && betaNoUs <= 1.0))
    throw std::invalid_argument("RescorlaWagner: learning rates must lie in (0, 1]");
}

double RescorlaWagner::response(const std::vector<std::size_t>& present) const {
  double total = 0.0;
  for (auto i : present) total += v_.at(i);
  return total;
}

void RescorlaWagner::trial(const std::vector<std::size_t>& present, bool usPresent) {
  const double error = (usPresent ? lambda_ : 0.0) - response(present);
  const double beta = usPresent ? betaUs_ : betaNoUs_;
  for (auto i : present) v_.at(i) += stimuli_[i].salience * beta * error;
}

std::size_t RescorlaWagner::index(const std::string& name) const {
  for (std::size_t i = 0; i < stimuli_.size(); ++i)
    if (stimuli_[i].name == name) return i;
  throw std::invalid_argument("RescorlaWagner: unknown stimulus " + name);
}

double acquisitionCurve(double alpha, double beta, double lambda, int n) {
  return lambda * (1.0 - std::pow(1.0 - alpha * beta, n));
}

double generalisedResponse(double v, double distance, double width) {
  if (width <= 0.0) throw std::invalid_argument("generalisedResponse: width must be positive");
  return v * std::exp(-std::fabs(distance) / width);
}

double generalisedResponse(const EllipticCGA& g, double v, const Multivector& conditioned, const Multivector& probe, double width) {
  return generalisedResponse(v, g.angle(conditioned, probe), width);
}

ConditioningRun runConditioning(RescorlaWagner model, const std::vector<ConditioningPhase>& phases) {
  ConditioningRun run;
  for (const auto& s : model.stimuli()) run.stimuli.push_back(s.name);
  for (const auto& phase : phases) {
    run.phases.push_back(phase.label);
    run.phaseStart.push_back(static_cast<int>(run.strengths.size()));
    for (int k = 0; k < phase.trials; ++k) {
      model.trial(phase.present, phase.usPresent);
      run.strengths.push_back(model.strengths());
    }
  }
  return run;
}

ConditioningRun pavlovExperiment(double salience, int acquisition, int extinction) {
  RescorlaWagner model({{"bell", salience}});
  return runConditioning(model, {{"bell + food", {0}, true, acquisition}, {"bell alone", {0}, false, extinction}});
}

BlockingResult blockingExperiment(int pretraining, int compound) {
  const std::vector<ConditionedStimulus> cs = {{"light", 0.3}, {"tone", 0.3}};
  BlockingResult r;
  r.blocked = runConditioning(RescorlaWagner(cs),
                              {{"light → food", {0}, true, pretraining}, {"light + tone → food", {0, 1}, true, compound}});
  r.control = runConditioning(RescorlaWagner(cs), {{"light + tone → food", {0, 1}, true, compound}});
  r.blockedTone = r.blocked.strengths.back()[1];
  r.controlTone = r.control.strengths.back()[1];
  return r;
}

FearConditioningResult fearConditioningExperiment(int pairings, double width) {
  FearConditioningResult r;
  r.run = runConditioning(RescorlaWagner({{"white rat", 0.4}}, 0.8), {{"rat + loud noise", {0}, true, pairings}});
  const double fear = r.run.strengths.back()[0];
  // Stimuli as points of the sphere S² of stimuli, placed at their psychological distance from
  // the white rat (furry, white, animate...) in different directions; the response reads the
  // distance back from the CGA inner product.
  const EllipticCGA g(2);
  const std::vector<double> rat = {0.0, 0.0, 1.0};
  const Multivector conditioned = g.point(rat);
  const std::vector<std::tuple<std::string, double, double>> probes = {
      {"white rat", 0.0, 0.0}, {"white rabbit", 0.4, 0.3}, {"dog", 0.8, 1.9}, {"fur coat", 1.0, 3.1}, {"cotton wool", 1.4, 4.4}, {"wooden blocks", 3.0, 5.5}};
  for (const auto& [name, d, azimuth] : probes) {
    const Multivector probe = d == 0.0 ? conditioned : g.pointAt(rat, {std::cos(azimuth), std::sin(azimuth), 0.0}, d);
    r.probes.push_back({name, g.angle(conditioned, probe), generalisedResponse(g, fear, conditioned, probe, width)});
  }
  return r;
}

}  // namespace syn
