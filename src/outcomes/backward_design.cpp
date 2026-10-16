#include "syn/outcomes/backward_design.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "syn/core/math.hpp"
#include "syn/core/random.hpp"
#include "syn/social/zpd.hpp"

namespace syn {

const char* contentKindName(ContentKind k) {
  switch (k) {
    case ContentKind::BigIdea: return "big idea";
    case ContentKind::Basic: return "basic idea";
    case ContentKind::NiceToKnow: return "worth being familiar with";
  }
  return "?";
}

std::vector<ContentItem> librarySessionContent() {
  using B = BloomLevel;
  using K = ContentKind;
  return {
      {"searching as strategic exploration", K::BigIdea, -1, B::Analyze, 10.0},
      {"Boolean operators", K::Basic, 0, B::Apply, 8.0},
      {"nesting and truncation", K::Basic, 0, B::Apply, 7.0},
      {"credible sources", K::BigIdea, -1, B::Evaluate, 10.0},
      {"authority", K::Basic, 3, B::Evaluate, 8.0},
      {"currency and accuracy", K::Basic, 3, B::Evaluate, 7.0},
      {"organization of information", K::BigIdea, -1, B::Understand, 9.0},
      {"keyword vs subject searching", K::Basic, 6, B::Understand, 7.0},
      {"classification and call numbers", K::Basic, 6, B::Understand, 6.0},
      {"catalogue limiters", K::Basic, 6, B::Apply, 6.0},
      {"library website tour", K::NiceToKnow, -1, B::Remember, 6.0},
      {"interlibrary loan form", K::NiceToKnow, -1, B::Remember, 4.0},
      {"chat reference link", K::NiceToKnow, -1, B::Remember, 2.0},
      {"Melvil Dewey", K::NiceToKnow, -1, B::Remember, 2.0},
      {"the term OPAC", K::NiceToKnow, -1, B::Remember, 2.0},
  };
}

std::vector<int> outcomePriority(const std::vector<ContentItem>& items) {
  std::vector<int> order;
  for (int j = 0; j < static_cast<int>(items.size()); ++j) {
    if (items[j].kind != ContentKind::BigIdea) continue;
    for (int i = 0; i < static_cast<int>(items.size()); ++i)
      if (items[i].kind == ContentKind::Basic && items[i].bigIdea == j) order.push_back(i);
    order.push_back(j);
  }
  for (int i = 0; i < static_cast<int>(items.size()); ++i)
    if (std::find(order.begin(), order.end(), i) == order.end()) order.push_back(i);
  return order;
}

namespace {
// Shares `minutes` among the chosen items in proportion to the time each needs.
std::vector<double> proportional(const std::vector<ContentItem>& items, const std::vector<int>& chosen, double minutes) {
  std::vector<double> t(items.size(), 0.0);
  double need = 0.0;
  for (int i : chosen) need += items[i].minutesToMaster;
  for (int i : chosen) t[i] = need > 0.0 ? minutes * items[i].minutesToMaster / need : 0.0;
  return t;
}
}  // namespace

SessionPlan coveragePlan(const std::vector<ContentItem>& items, double minutes) {
  SessionPlan p;
  p.name = "coverage-based";
  p.minutes.assign(items.size(), minutes / static_cast<double>(items.size()));
  p.activeShare = 0.05;  // a demonstration, then a rigid lecture
  p.alignment = 0.3;
  return p;
}

SessionPlan activityPlan(const std::vector<ContentItem>& items, double minutes) {
  SessionPlan p;
  p.name = "activity-based";
  std::vector<int> chosen;
  for (int i = 0; i < static_cast<int>(items.size()); ++i)
    if (items[i].kind == ContentKind::Basic && items[i].level == BloomLevel::Apply) chosen.push_back(i);  // hands-on searching
  p.minutes = proportional(items, chosen, minutes);
  p.activeShare = 0.85;
  p.alignment = 0.15;  // practice without the "when" and "why"
  return p;
}

SessionPlan backwardDesignPlan(const std::vector<ContentItem>& items, double minutes, int outcomes) {
  if (outcomes < 1) throw std::invalid_argument("backwardDesignPlan: need at least one outcome");
  const auto order = outcomePriority(items);
  const std::vector<int> chosen(order.begin(), order.begin() + std::min<std::size_t>(order.size(), static_cast<std::size_t>(outcomes)));
  SessionPlan p;
  p.name = "backward design (" + std::to_string(outcomes) + " outcomes)";
  p.minutes = proportional(items, chosen, minutes);
  p.activeShare = 0.6;
  p.alignment = 0.9;
  p.outcomesShared = true;
  p.essentialQuestions = 0.7;
  p.affective = 0.5;
  p.outcomes = static_cast<int>(chosen.size());
  return p;
}

SessionResult simulateSession(const std::vector<ContentItem>& items, const SessionPlan& plan, const SessionParams& p,
                              std::uint64_t seed) {
  const std::size_t n = items.size();
  if (n == 0 || plan.minutes.size() != n || p.learners < 1) throw std::invalid_argument("simulateSession: bad sizes");
  SessionResult r;
  r.plan = plan.name;
  r.mastery.assign(n, 0.0);
  std::vector<int> touched;
  for (std::size_t i = 0; i < n; ++i)
    if (plan.minutes[i] > 0.0) touched.push_back(static_cast<int>(i));
  r.itemsTouched = static_cast<int>(touched.size());
  r.coverage = static_cast<double>(touched.size()) / static_cast<double>(n);
  const double efficiency = 0.45 + 0.55 * clamp01(plan.activeShare);
  const double crowding = 1.0 / (1.0 + p.overload * std::max(0.0, r.itemsTouched - p.capacity));
  const double relevance = 0.5 * plan.alignment + 0.3 * plan.essentialQuestions + 0.2 * plan.affective + (plan.outcomesShared ? 0.1 : 0.0);
  const double judgeBias = plan.outcomesShared ? 0.0 : 0.2, judgeNoise = plan.outcomesShared ? 0.12 : 0.3;
  int bigCount = 0, basicCount = 0, niceCount = 0;
  for (const auto& it : items) {
    bigCount += it.kind == ContentKind::BigIdea;
    basicCount += it.kind == ContentKind::Basic;
    niceCount += it.kind == ContentKind::NiceToKnow;
  }
  const double L = p.learners;
  std::vector<double> m(n), u(n);
  for (int l = 0; l < p.learners; ++l) {
    Rng rng = Rng::stream(seed, static_cast<std::uint64_t>(l));
    const double aptitude = std::exp(p.aptitudeSd * rng.normal());
    for (std::size_t i = 0; i < n; ++i) {
      const double t = std::max(0.0, plan.minutes[i] - (plan.minutes[i] > 0.0 ? p.transitionMinutes : 0.0));
      m[i] = 1.0 - std::exp(-aptitude * p.rate * t * efficiency * crowding / items[i].minutesToMaster);
      r.mastery[i] += m[i] / L;
    }
    // Big ideas rest on their basic ideas.
    double transferBest = 0.0, transferSum = 0.0;
    for (std::size_t j = 0; j < n; ++j) {
      if (items[j].kind != ContentKind::BigIdea) continue;
      double s = 0.0;
      int c = 0;
      for (std::size_t i = 0; i < n; ++i)
        if (items[i].kind == ContentKind::Basic && items[i].bigIdea == static_cast<int>(j)) {
          s += m[i];
          ++c;
        }
      u[j] = m[j] * (c ? s / c : 1.0);
      const double tr = u[j] * (0.3 + 0.7 * clamp01(plan.alignment));
      transferSum += tr;
      transferBest = std::max(transferBest, tr);
      r.bigIdeas += u[j] / (L * bigCount);
    }
    for (std::size_t i = 0; i < n; ++i) {
      if (items[i].kind == ContentKind::Basic) r.basics += m[i] / (L * basicCount);
      if (items[i].kind == ContentKind::NiceToKnow) r.niceToKnow += m[i] / (L * niceCount);
      if (items[i].kind == ContentKind::Basic) r.enduring += m[i] / L;
      if (items[i].kind == ContentKind::BigIdea) r.enduring += 2.0 * u[i] / L;
    }
    r.transfer += bigCount ? transferSum / (L * bigCount) : 0.0;
    // Do they use it afterwards, or go back to their old habits?
    r.behaviourChange += rng.bernoulli(logistic(6.0 * (transferBest - 0.25) + 2.0 * (relevance - 0.5))) ? 1.0 / L : 0.0;
    // Outcomes met, and learners' own judgement of whether they met them.
    for (int i : touched) {
      const bool met = m[i] >= p.threshold;
      const bool judged = m[i] + judgeBias + judgeNoise * rng.normal() >= p.threshold;
      r.outcomesMet += met ? 1.0 / (L * touched.size()) : 0.0;
      r.selfAssessment += met == judged ? 1.0 / (L * touched.size()) : 0.0;
    }
  }
  return r;
}

OutcomeLevelFit fitOutcomeLevel(double audience, double scaffold, double baseDifficulty, double step) {
  OutcomeLevelFit f;
  ZpdParams zp;
  zp.scaffold = scaffold;
  int lastIndependent = -1;
  f.recommended = -1;
  for (int k = 0; k < kBloomLevels; ++k) {
    const ZpdAssessment a = assessTask(audience, baseDifficulty + step * k, zp);
    f.alone.push_back(a.alone);
    f.assisted.push_back(a.assisted);
    f.gain.push_back(zp.learningRate * a.assisted * (1.0 - a.alone));
    f.zone.push_back(static_cast<int>(a.zone));
    if (a.zone == Zone::Proximal) f.recommended = k;
    if (a.zone == Zone::Independent) lastIndependent = k;
  }
  if (f.recommended < 0) f.recommended = std::max(lastIndependent, 0);
  return f;
}

}  // namespace syn
