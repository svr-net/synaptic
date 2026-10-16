#include "syn/assessment/session.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "syn/core/math.hpp"
#include "syn/core/random.hpp"
#include "syn/social/zpd.hpp"

namespace syn {

const char* formativeCheckName(FormativeCheck c) {
  switch (c) {
    case FormativeCheck::None: return "none";
    case FormativeCheck::Informal: return "informal check";
    case FormativeCheck::Formal: return "formal check";
  }
  return "?";
}

const char* cycleEvidenceName(CycleEvidence e) {
  switch (e) {
    case CycleEvidence::None: return "no assessment";
    case CycleEvidence::Grades: return "grades";
    case CycleEvidence::SelfReport: return "self-report (indirect)";
    case CycleEvidence::Direct: return "direct assessment";
  }
  return "?";
}

namespace {

constexpr double kInformalMinutes = 1.0, kFormalMinutes = 2.0, kSummativeMinutes = 5.0;

// Support from t minutes on a chunk, relative to the nominal time: diminishing returns.
double support(double relativeMinutes) { return (1.0 - std::exp(-2.0 * relativeMinutes)) / (1.0 - std::exp(-2.0)); }

// One class: per-chunk mean mastery, the share meeting the outcome and the instructor's view.
struct ClassRun {
  std::vector<double> chunkMean;
  double finalMean = 0.0, met = 0.0, estimatedMet = 0.0, assessmentMinutes = 0.0;
  std::vector<double> chunkMet;  // share of learners meeting the threshold on each chunk
  double lastCheckedMet = 0.0;   // share meeting the threshold on the last chunk a formative check sees
  int reteaches = 0, missedGaps = 0;
};

ClassRun runClass(const AssessedSessionParams& p, const std::vector<double>& clarity, Rng& rng) {
  const int n = p.classSize, J = p.chunks;
  const double mu = p.betweenClassSd * rng.normal();
  std::vector<double> theta(n);
  for (double& t : theta) t = mu + p.withinClassSd * rng.normal();
  const double checkMinutes = p.formative == FormativeCheck::Formal ? kFormalMinutes : p.formative == FormativeCheck::Informal ? kInformalMinutes : 0.0;
  ClassRun r;
  const bool checking = p.formative != FormativeCheck::None;
  r.assessmentMinutes = checkMinutes * J + (p.summative ? kSummativeMinutes : 0.0);
  const double nominal = p.minutes / J;
  double budget = p.minutes - r.assessmentMinutes;  // teaching minutes left
  // An instructor who checks keeps one reteach in hand; unused, it becomes practice at the end.
  double reserve = checking ? std::min(p.reteachMinutes, budget / (J + 1)) : 0.0;
  budget -= reserve;
  std::vector<double> prev(n, 1.0), m(n);
  ZpdParams zp;
  std::vector<double> c = clarity;
  for (int j = 0; j < J; ++j) {
    const double b = p.difficulty - 0.3 * (J - 1 - j);
    double minutes = budget / (J - j);
    budget -= minutes;
    auto teach = [&](double s0) {
      zp.scaffold = p.scaffold * s0;
      double s = 0.0;
      for (int i = 0; i < n; ++i) {
        m[i] = assessTask(theta[i] + c[j] + p.prerequisite * (prev[i] - 1.0), b, zp).assisted;
        s += m[i];
      }
      return s / n;
    };
    double mean = teach(support(minutes / nominal));
    if (checking) {
      // Taking the temperature of the class before building on the chunk.
      const bool informal = p.formative == FormativeCheck::Informal;
      const double estimate = mean + (informal ? 0.15 : 0.0) + (informal ? 0.10 : 0.04) * rng.normal();
      double extra = 0.0;
      if (estimate >= p.reteachThreshold && mean < p.reteachThreshold) ++r.missedGaps;
      if (estimate < p.reteachThreshold) {
        extra = reserve > 0.0 ? reserve : std::min(p.reteachMinutes, budget * 0.5);
        if (reserve > 0.0) reserve = 0.0;
        else budget -= extra;
        if (extra > 0.0) {
          c[j] += p.reteachClarity * extra / p.reteachMinutes;
          ++r.reteaches;
        }
      } else if (j == J - 1 && reserve > 0.0) {
        extra = reserve;  // the reteach was not needed: more practice
      }
      if (extra > 0.0) {
        minutes += extra;
        mean = teach(support(minutes / nominal));
      }
    }
    r.chunkMean.push_back(mean);
    double met = 0.0;
    for (double x : m) met += (x >= p.threshold ? 1.0 : 0.0) / n;
    r.chunkMet.push_back(met);
    if (j == J - 1) r.lastCheckedMet = met;
    prev = m;
  }
  r.finalMean = r.chunkMean.back();
  for (double x : m) r.met += (x >= p.threshold ? 1.0 : 0.0) / n;
  // The instructor's view of whether the outcome was met.
  if (p.summative) {
    double est = 0.0;
    for (double x : m) est += (x + 0.12 * rng.normal() >= p.threshold ? 1.0 : 0.0) / n;
    r.estimatedMet = est;
  } else if (p.formative == FormativeCheck::Formal) {
    // The check on the last chunk, before the closing practice: close, but not the end result.
    r.estimatedMet = clamp01(r.lastCheckedMet + 0.04 * rng.normal());
  } else if (p.formative == FormativeCheck::Informal) {
    r.estimatedMet = clamp01(r.lastCheckedMet + 0.15 + 0.10 * rng.normal());  // learners nod along
  } else {
    r.estimatedMet = 0.85;  // "they seemed to get it"
  }
  return r;
}

std::vector<double> drawClarity(const AssessedSessionParams& p, Rng& rng) {
  std::vector<double> c(p.chunks);
  for (double& x : c) x = p.clarityMean + p.claritySd * rng.normal();
  return c;
}

void validate(const AssessedSessionParams& p) {
  if (p.classes < 1 || p.classSize < 1 || p.chunks < 2 || p.minutes <= 0.0) throw std::invalid_argument("assessed session: bad sizes");
  if (!p.chunkClarity.empty() && static_cast<int>(p.chunkClarity.size()) != p.chunks)
    throw std::invalid_argument("assessed session: chunkClarity must have one value per chunk");
}

}  // namespace

AssessedSessionResult simulateAssessedSession(const AssessedSessionParams& p, std::uint64_t seed) {
  validate(p);
  AssessedSessionResult out;
  out.name = std::string(formativeCheckName(p.formative)) + (p.summative ? " + summative" : "");
  out.chunkMastery.assign(p.chunks, 0.0);
  const double C = p.classes;
  for (int k = 0; k < p.classes; ++k) {
    Rng rng = Rng::stream(seed, static_cast<std::uint64_t>(k));
    const auto clarity = p.chunkClarity.empty() ? drawClarity(p, rng) : p.chunkClarity;
    const auto r = runClass(p, clarity, rng);
    for (int j = 0; j < p.chunks; ++j) out.chunkMastery[j] += r.chunkMean[j] / C;
    out.finalMastery += r.finalMean / C;
    out.outcomeMet += r.met / C;
    out.reteaches += r.reteaches / C;
    out.missedGaps += r.missedGaps / C;
    out.assessmentMinutes += r.assessmentMinutes / C;
    out.estimateError += std::fabs(r.estimatedMet - r.met) / C;
  }
  return out;
}

AssessmentCycleResult simulateAssessmentCycle(const AssessmentCycleParams& cp, std::uint64_t seed) {
  AssessedSessionParams p = cp.session;
  validate(p);
  if (cp.offerings < 1) throw std::invalid_argument("assessment cycle: need offerings");
  if (p.chunkClarity.empty()) {
    Rng rng = Rng::stream(seed, 0xC1A7);
    p.chunkClarity = drawClarity(p, rng);
  }
  AssessmentCycleResult out;
  out.evidence = cycleEvidenceName(cp.evidence);
  for (int o = 0; o < cp.offerings; ++o) {
    // Collect data: run the offering (fresh classes each time, same instructor).
    std::vector<double> chunk(p.chunks, 0.0);
    double met = 0.0, mastery = 0.0;
    const double C = p.classes;
    for (int k = 0; k < p.classes; ++k) {
      Rng rng = Rng::stream(seed, static_cast<std::uint64_t>(o) * 1000003ull + static_cast<std::uint64_t>(k));
      const auto r = runClass(p, p.chunkClarity, rng);
      for (int j = 0; j < p.chunks; ++j) chunk[j] += r.chunkMet[j] / C;
      met += r.met / C;
      mastery += r.finalMean / C;
    }
    out.met.push_back(met);
    out.mastery.push_back(mastery);
    out.difficulty.push_back(p.difficulty);
    out.clarity.push_back(mean(p.chunkClarity));
    // Analyse: what the evidence says about each chunk and about the outcome.
    Rng noise = Rng::stream(seed ^ 0x5EEDull, static_cast<std::uint64_t>(o));
    std::vector<double> measured(p.chunks);
    double estMet = 0.0;
    switch (cp.evidence) {
      case CycleEvidence::None: estMet = -1.0; break;
      case CycleEvidence::Direct:
        for (int j = 0; j < p.chunks; ++j) measured[j] = chunk[j] + 0.03 * noise.normal();
        estMet = met + 0.03 * noise.normal();
        break;
      case CycleEvidence::SelfReport:
        // Learners regress to a flattering 0.6 and add a little.
        for (int j = 0; j < p.chunks; ++j) measured[j] = chunk[j] + 0.6 * (0.8 - chunk[j]) + 0.05 + 0.03 * noise.normal();
        estMet = clamp01(met + 0.6 * (0.75 - met) + 0.05 + 0.03 * noise.normal());
        break;
      case CycleEvidence::Grades:
        // A "B" for attending and taking part: about the class, not about any chunk.
        for (int j = 0; j < p.chunks; ++j) measured[j] = 0.55 + 0.35 * mastery + 0.04 * noise.normal();
        estMet = clamp01(0.55 + 0.4 * met + 0.05 * noise.normal());
        break;
    }
    out.estimatedMet.push_back(estMet);
    if (cp.evidence == CycleEvidence::None) continue;
    // Revisit outcomes and reflect on instruction.
    // Revising a chunk helps as far as it was actually weak; rewriting a sound one gains little.
    for (int j = 0; j < p.chunks; ++j)
      if (measured[j] < cp.gapThreshold) p.chunkClarity[j] += cp.revise * clamp01((0.85 - chunk[j]) / 0.3);
    if (estMet > cp.raiseAbove) p.difficulty += cp.step;
    else if (estMet < cp.lowerBelow) p.difficulty -= cp.step;
  }
  return out;
}

}  // namespace syn
