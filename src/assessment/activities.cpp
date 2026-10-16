#include "syn/assessment/activities.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "syn/core/math.hpp"
#include "syn/core/random.hpp"

namespace syn {

std::vector<AssessmentActivity> AssessmentActivity::catalogue() {
  using B = BloomLevel;
  // name, minutes, direct, formal, authentic, measures, noise, reach, writing, testLike, afterSession
  return {
      {"quick poll (\"got it?\")", 1.0, false, false, false, B::Remember, 0.30, 0.8, 0.0, 0.1, false},
      {"observing practice", 0.0, true, false, true, B::Apply, 0.30, 0.4, 0.0, 0.0, false},
      {"worksheet", 10.0, true, true, true, B::Apply, 0.12, 1.0, 0.4, 0.3, false},
      {"task / demonstration", 8.0, true, true, true, B::Apply, 0.15, 0.6, 0.0, 0.2, false},
      {"pre/post test (fixed choice)", 8.0, true, true, false, B::Remember, 0.12, 1.0, 0.1, 1.0, false},
      {"graphic organizer", 6.0, true, true, false, B::Understand, 0.15, 1.0, 0.2, 0.3, false},
      {"pro/con chart", 8.0, true, true, true, B::Analyze, 0.18, 1.0, 0.4, 0.2, false},
      {"annotated bibliography", 0.0, true, true, true, B::Evaluate, 0.10, 0.6, 0.9, 0.4, true},
      {"exit ticket", 3.0, true, true, false, B::Understand, 0.22, 1.0, 0.4, 0.2, false},
      {"directed paraphrasing", 4.0, true, true, true, B::Understand, 0.18, 1.0, 0.6, 0.2, false},
      {"one-sentence summary", 3.0, true, true, false, B::Understand, 0.22, 1.0, 0.5, 0.2, false},
      {"application card", 4.0, true, true, true, B::Apply, 0.22, 1.0, 0.5, 0.2, false},
      {"self-assessment questionnaire", 3.0, false, true, false, B::Remember, 0.10, 1.0, 0.2, 0.3, false},
  };
}

std::vector<AssessmentAudience> AssessmentAudience::all() {
  return {
      {"undergraduates", 0.7, 0.6},
      {"high school students", 0.5, 0.8},
      {"older adults (public library)", 0.4, 0.1},
      {"English-language learners", 0.2, 0.5},
  };
}

std::vector<ActivityScore> scoreActivities(BloomLevel outcome, const AssessmentAudience& audience, double sessionMinutes) {
  if (sessionMinutes <= 0.0) throw std::invalid_argument("scoreActivities: need a positive session length");
  std::vector<ActivityScore> out;
  const int o = static_cast<int>(outcome);
  for (const auto& a : AssessmentActivity::catalogue()) {
    ActivityScore s;
    s.activity = a.name;
    const int m = static_cast<int>(a.measures);
    // Evidence below the outcome's level overstates it; above, it asks for more than was taught.
    s.alignment = m == o ? 1.0 : m < o ? std::pow(0.4, o - m) : std::pow(0.75, m - o);
    if (!a.direct) s.alignment *= 0.3;  // learners tell, they do not show
    s.information = a.reach * (1.0 - a.noise);
    s.fit = clamp01(1.0 - 1.2 * std::max(0.0, a.writingLoad - audience.writing) - 1.2 * std::max(0.0, a.testLike - audience.tests));
    s.timeCost = a.minutes / sessionMinutes;
    // A sixth of the session is a fair price; beyond that the activity eats the teaching.
    s.value = s.alignment * s.information * s.fit - std::max(0.0, s.timeCost - 0.15);
    out.push_back(s);
  }
  return out;
}

namespace {
double correlation(const std::vector<double>& x, const std::vector<double>& y) {
  const double mx = mean(x), my = mean(y);
  double sxy = 0.0, sxx = 0.0, syy = 0.0;
  for (std::size_t i = 0; i < x.size(); ++i) {
    sxy += (x[i] - mx) * (y[i] - my);
    sxx += (x[i] - mx) * (x[i] - mx);
    syy += (y[i] - my) * (y[i] - my);
  }
  return sxx > 0.0 && syy > 0.0 ? sxy / std::sqrt(sxx * syy) : 0.0;
}
}  // namespace

SignalValidity simulateSignals(int learners, std::uint64_t seed) {
  if (learners < 3) throw std::invalid_argument("simulateSignals: need learners");
  std::vector<double> l, direct, grade, self, satisfaction;
  for (int i = 0; i < learners; ++i) {
    Rng rng = Rng::stream(seed, static_cast<std::uint64_t>(i));
    const double learning = rng.normal();
    const double attendance = rng.normal(), effort = 0.3 * learning + 0.95 * rng.normal(), conduct = rng.normal();
    const double skill = logistic(learning);
    l.push_back(learning);
    direct.push_back(learning + 0.45 * rng.normal());
    grade.push_back(0.45 * learning + 0.3 * attendance + 0.35 * effort + 0.2 * conduct + 0.3 * rng.normal());
    // Self-report regresses to a flattering mean: the least skilled overestimate most.
    self.push_back(skill + 0.6 * (0.6 - skill) + 0.05 + 0.12 * rng.normal());
    satisfaction.push_back(0.25 * learning + rng.normal());
  }
  return {correlation(l, direct), correlation(l, grade), correlation(l, self), correlation(l, satisfaction)};
}

}  // namespace syn
