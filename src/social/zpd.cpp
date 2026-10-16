#include "syn/social/zpd.hpp"

#include <cmath>
#include <stdexcept>

#include "syn/core/elliptic_cga.hpp"
#include "syn/core/math.hpp"
#include "syn/core/random.hpp"

namespace syn {

const char* zoneName(Zone z) {
  switch (z) {
    case Zone::Independent: return "can do alone";
    case Zone::Proximal: return "zone of proximal development";
    case Zone::Beyond: return "cannot do yet";
  }
  return "?";
}

const char* taskPolicyName(TaskPolicy p) {
  switch (p) {
    case TaskPolicy::TooEasy: return "too easy (review)";
    case TaskPolicy::ClassAverage: return "one level for the whole class";
    case TaskPolicy::TooHard: return "too hard";
    case TaskPolicy::Zpd: return "targeted at each learner's ZPD";
  }
  return "?";
}

namespace {

// The circle of tasks around a learner: Cl(2, 1), learner at the pole n, tasks along m.
const EllipticCGA& circle() {
  static const EllipticCGA g(1);
  return g;
}
const std::vector<double> kPole = {1.0, 0.0};

const Multivector& learnerPoint() {
  static const Multivector l = circle().point(kPole);
  return l;
}

// The task as a null vector at angle φ from the learner: cos φ·n + sin φ·m + e.
Multivector taskPoint(double theta, double b) {
  const double phi = taskAngle(b - theta);
  return Multivector::vector(2, 1, {std::cos(phi), std::sin(phi), 1.0});
}

// The scaffold as a dilation versor: Δ ↦ Δ − s.
Multivector scaffolded(const Multivector& task, double scaffold) {
  return circle().move(circle().dilation(kPole, -scaffold / 2.0), task);
}

double successOf(const Multivector& task) { return 1.0 + 0.5 * circle().dot(task, learnerPoint()); }

// The cap of points the learner succeeds on with probability at least p: L − 2(1 − p)·e,
// since 1 − cos ρ = 2(1 − p).
Multivector zoneCap(double p) { return learnerPoint() - circle().e() * (2.0 * (1.0 - p)); }

}  // namespace

double taskAngle(double relativeDifficulty) { return 2.0 * std::atan(std::exp(relativeDifficulty / 2.0)); }

double successAlone(double theta, double b) { return successOf(taskPoint(theta, b)); }

double successAssisted(double theta, double b, double scaffold) { return successOf(scaffolded(taskPoint(theta, b), scaffold)); }

ZpdAssessment assessTask(double theta, double b, const ZpdParams& p) {
  const Multivector t = taskPoint(theta, b), helped = scaffolded(t, p.scaffold);
  ZpdAssessment a;
  a.alone = successOf(t);
  a.assisted = successOf(helped);
  a.zone = circle().inside(t, zoneCap(p.independentThreshold))      ? Zone::Independent
           : circle().inside(helped, zoneCap(p.assistedThreshold)) ? Zone::Proximal
                                                                   : Zone::Beyond;
  return a;
}

Zone zoneOf(double theta, double b, const ZpdParams& p) { return assessTask(theta, b, p).zone; }

double expectedGain(double theta, double b, const ZpdParams& p) {
  const ZpdAssessment a = assessTask(theta, b, p);
  return p.learningRate * a.assisted * (1.0 - a.alone);
}

std::pair<double, double> zpdBand(double theta, const ZpdParams& p) {
  // A cap of radius ρ around the learner holds the tasks with Δ ≤ 2·ln tan(ρ/2); the scaffold
  // shifts the outer boundary out by s.
  const double rhoHi = std::acos(2.0 * p.independentThreshold - 1.0), rhoLo = std::acos(2.0 * p.assistedThreshold - 1.0);
  return {theta + 2.0 * std::log(std::tan(rhoHi / 2.0)), theta + p.scaffold + 2.0 * std::log(std::tan(rhoLo / 2.0))};
}

std::pair<double, double> zpdMapRadii(const ZpdParams& p) {
  const auto band = zpdBand(0.0, p);
  return {std::exp(band.first / 2.0), std::exp(band.second / 2.0)};
}

std::vector<ZpdMapPoint> zpdMap(const ZpdParams& p, int rings, int perRing, double minDifficulty, double maxDifficulty) {
  if (rings < 2 || perRing < 1) throw std::invalid_argument("zpdMap: bad sizes");
  const EllipticCGA sphere(2);
  const std::vector<double> pole = {0.0, 0.0, 1.0};
  const Multivector learner = sphere.point(pole);
  const Multivector independent = sphere.cap(learner, std::acos(2.0 * p.independentThreshold - 1.0));
  const Multivector assisted = sphere.cap(learner, std::acos(2.0 * p.assistedThreshold - 1.0));
  const Multivector scaffold = sphere.dilation(pole, -p.scaffold / 2.0);
  std::vector<ZpdMapPoint> out;
  for (int r = 0; r < rings; ++r) {
    // Rings evenly spaced in the plane: radius tan(φ/2) = e^{Δ/2} from that of the easiest
    // task to that of the hardest.
    const double rMin = std::exp(minDifficulty / 2.0), rMax = std::exp(maxDifficulty / 2.0);
    const double delta = 2.0 * std::log(rMin + (rMax - rMin) * r / (rings - 1));
    for (int k = 0; k < perRing; ++k) {
      const double az = 2.0 * M_PI * (k + 0.5 * (r % 2)) / perRing;
      const Multivector t = sphere.pointAt(pole, {std::cos(az), std::sin(az), 0.0}, taskAngle(delta));
      ZpdMapPoint m;
      const auto xy = sphere.projectStereographic(t);
      m.x = xy[0];
      m.y = xy[1];
      m.difficulty = delta;
      m.zone = sphere.inside(t, independent) ? Zone::Independent
               : sphere.inside(sphere.move(scaffold, t), assisted) ? Zone::Proximal
                                                                   : Zone::Beyond;
      out.push_back(m);
    }
  }
  return out;
}

TutoringRun simulateTutoring(const ZpdParams& zpd, const TutoringParams& tp, TaskPolicy policy, std::uint64_t seed) {
  if (tp.learners < 1 || tp.sessions < 1) throw std::invalid_argument("simulateTutoring: bad sizes");
  TutoringRun run;
  run.policy = taskPolicyName(policy);
  const auto n = static_cast<std::size_t>(tp.sessions);
  run.zoneShare.assign(3, std::vector<double>(n, 0.0));
  std::vector<std::vector<double>> ability(n, std::vector<double>(tp.learners));
  const auto band = zpdBand(0.0, zpd);
  const double zpdCentre = 0.5 * (band.first + band.second);  // offset of the ZPD's middle from θ
  for (int i = 0; i < tp.learners; ++i) {
    Rng rng = Rng::stream(seed, static_cast<std::uint64_t>(i));
    double theta = rng.normal(tp.abilityMean, tp.abilitySd);
    const double start = theta;
    for (std::size_t s = 0; s < n; ++s) {
      const double estimate = theta + tp.assessmentNoise * rng.normal();
      const double classLevel = tp.classDifficulty + tp.curriculumPace * static_cast<double>(s);
      double b = classLevel;
      switch (policy) {
        case TaskPolicy::TooEasy: b = theta - 2.5; break;
        case TaskPolicy::ClassAverage: b = classLevel; break;
        case TaskPolicy::TooHard: b = theta + 5.0; break;
        case TaskPolicy::Zpd: b = estimate + zpdCentre; break;
      }
      ZpdParams here = zpd;
      if (tp.fadeScaffold) here.scaffold = zpd.scaffold * (0.4 + 0.6 * (1.0 - successAlone(theta, b)));
      ability[s][i] = theta;
      run.zoneShare[static_cast<int>(zoneOf(theta, b, zpd))][s] += 1.0 / tp.learners;
      theta += expectedGain(theta, b, here) * (1.0 + 0.2 * rng.normal());
    }
    run.finalGain += (theta - start) / tp.learners;
  }
  for (const auto& a : ability) {
    run.meanAbility.push_back(mean(a));
    run.q10Ability.push_back(quantile(a, 0.1));
    run.q90Ability.push_back(quantile(a, 0.9));
  }
  return run;
}

}  // namespace syn
