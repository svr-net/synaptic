#pragma once

// Classical (stimulus) conditioning: Pavlov's bell and Watson's experiment, modelled with the
// Rescorla–Wagner rule. Each conditioned stimulus (CS) carries an associative strength V;
// on a trial every present CS moves by ΔV = α·β·(λ − ΣV_present), where λ is 1 when the
// unconditioned stimulus (US, e.g. food or a loud noise) follows and 0 when it does not.

#include <cstddef>
#include <string>
#include <vector>

namespace syn {

struct ConditionedStimulus {
  std::string name;
  double salience = 0.3;  // α in (0, 1]
};

class RescorlaWagner {
 public:
  /** betaUs: learning rate on trials with the US; betaNoUs: on trials without it (extinction). */
  RescorlaWagner(std::vector<ConditionedStimulus> stimuli, double betaUs = 0.5, double betaNoUs = 0.3,
                 double lambda = 1.0);

  /** One trial with the listed stimuli present; usPresent says whether the US follows. */
  void trial(const std::vector<std::size_t>& present, bool usPresent);

  /** Prediction of the US (conditioned response strength) for a compound of stimuli. */
  double response(const std::vector<std::size_t>& present) const;

  double strength(std::size_t i) const { return v_.at(i); }
  const std::vector<double>& strengths() const { return v_; }
  const std::vector<ConditionedStimulus>& stimuli() const { return stimuli_; }
  std::size_t index(const std::string& name) const;

 private:
  std::vector<ConditionedStimulus> stimuli_;
  std::vector<double> v_;
  double betaUs_, betaNoUs_, lambda_;
};

/** Closed form of acquisition for a single CS: V_n = λ·(1 − (1 − αβ)^n). */
double acquisitionCurve(double alpha, double beta, double lambda, int n);

/**
 * Stimulus generalisation (Shepard's exponential gradient): the response to a test stimulus at
 * psychological distance d from the conditioned one is V·exp(−d / width).
 */
double generalisedResponse(double v, double distance, double width);

/**
 * The same on the sphere of stimuli (pure elliptic CGA): stimuli are null vectors, and the
 * psychological distance is the geodesic angle between them, read off their inner product.
 */
class Multivector;
class EllipticCGA;
double generalisedResponse(const EllipticCGA& g, double v, const Multivector& conditioned, const Multivector& probe, double width);

struct ConditioningPhase {
  std::string label;
  std::vector<std::size_t> present;
  bool usPresent = true;
  int trials = 10;
};

struct ConditioningRun {
  std::vector<std::string> stimuli;
  std::vector<std::vector<double>> strengths;  // after each trial: [trial][stimulus]
  std::vector<std::string> phases;
  std::vector<int> phaseStart;  // first trial index of each phase
};

/** Runs the phases in order and records every stimulus' strength after each trial. */
ConditioningRun runConditioning(RescorlaWagner model, const std::vector<ConditioningPhase>& phases);

/** Pavlov: bell paired with food (acquisition), then the bell alone (extinction). */
ConditioningRun pavlovExperiment(double salience = 0.3, int acquisition = 30, int extinction = 30);

/**
 * Kamin blocking: light → food is learned first; then light + tone → food. The tone learns
 * little because the food is already predicted. A control group sees only the compound.
 */
struct BlockingResult {
  double blockedTone = 0.0;  // tone strength after pre-training on the light
  double controlTone = 0.0;  // tone strength without pre-training
  ConditioningRun blocked, control;
};
BlockingResult blockingExperiment(int pretraining = 20, int compound = 20);

/**
 * Watson's experiment: a white rat paired with a loud noise. Fear of the rat is acquired, then
 * generalises to stimuli that resemble it (rabbit, dog, fur coat) and not to unlike ones (blocks).
 */
struct GeneralisationProbe {
  std::string stimulus;
  double distance = 0.0;  // geodesic angle from the conditioned stimulus (radians)
  double fear = 0.0;
};
struct FearConditioningResult {
  ConditioningRun run;
  std::vector<GeneralisationProbe> probes;
};
FearConditioningResult fearConditioningExperiment(int pairings = 7, double width = 1.0);

}  // namespace syn
