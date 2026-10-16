#include "syn/classroom/classroom_engine.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>

#include "syn/core/random.hpp"
#include "syn/development/piaget.hpp"
#include "syn/humanism/maslow.hpp"
#include "syn/social/zpd.hpp"

namespace syn {

namespace {

constexpr int kReviewSlots = 3;
constexpr int kDrawsPerSession = 3 + 4 * kReviewSlots;

struct Unit {
  double learned = 0.0;  // share of the unit's content acquired, [0, 1]
  MemoryTrace trace;
  bool introduced = false;
};

// Everything a learner carries through the course.
struct State {
  Learner l;
  double habit = 0.0;           // behaviourist habit strength from reinforcement
  double discrimination = 0.0;  // ability to tell related problem types apart
  double persistence = 0.0;     // effort carried over from how the learner met past failures
  bool engaged = true;
  std::vector<Unit> units;
};

double choiceAccuracy(double d) { return 1.0 / 3.0 + 2.0 / 3.0 * (1.0 - std::exp(-(0.6 + d))); }

}  // namespace

ClassroomEngine::ClassroomEngine(PopulationSpec population, CourseSpec course, SimulationSettings settings, MemoryParams memory)
    : population_(std::move(population)), course_(course), settings_(settings), memory_(memory) {
  if (course_.sessions < 1 || course_.sessionGap <= 0.0 || course_.delayedTest < 0.0 || course_.reportStep <= 0.0)
    throw std::invalid_argument("CourseSpec: need sessions >= 1, sessionGap > 0, delayedTest >= 0, reportStep > 0");
  if (settings_.learners < 2) throw std::invalid_argument("SimulationSettings: need at least two learners");
}

ClassroomResult ClassroomEngine::run(const InstructionalDesign& design) const { return simulate(design, settings_.seed, true); }

ClassroomResult ClassroomEngine::run(const InstructionalDesign& design, std::uint64_t seed) const {
  return simulate(design, seed, true);
}

double ClassroomEngine::objective(const InstructionalDesign& design) const {
  return simulate(design, settings_.seed, false).delayedScore;
}

double ClassroomEngine::objective(const InstructionalDesign& design, std::uint64_t seed) const {
  return simulate(design, seed, false).delayedScore;
}

ClassroomResult ClassroomEngine::simulate(const InstructionalDesign& raw, std::uint64_t seed, bool profiles) const {
  const InstructionalDesign d = raw.clamped();
  const CourseSpec& c = course_;
  const int n = settings_.learners, sessions = c.sessions;
  const double lastSession = c.sessionGap * (sessions - 1);
  const double testDay = lastSession + c.delayedTest;
  const double endDay = lastSession + 1.0;
  const double finalLevel = c.difficulty + c.difficultyStep * (sessions - 1);

  // Reporting days: the grid plus the two test days, so both are reported exactly.
  std::vector<double> days;
  if (profiles) {
    for (double t = 0.0; t <= testDay + 1e-9; t += c.reportStep) days.push_back(t);
    days.push_back(endDay);
    days.push_back(testDay);
    std::sort(days.begin(), days.end());
    days.erase(std::unique(days.begin(), days.end(), [](double a, double b) { return std::fabs(a - b) < 1e-9; }), days.end());
  } else {
    days = {endDay, testDay};
  }
  const std::size_t nd = days.size();
  std::vector<std::vector<double>> score(nd, std::vector<double>(n)), recall(nd, std::vector<double>(n));

  ClassroomResult r;
  r.learnerCount = n;
  r.lastSessionDay = lastSession;
  r.delayedTestDay = testDay;
  auto& tr = r.sessions;
  for (auto* v : {&tr.ability, &tr.effort, &tr.motivation, &tr.value, &tr.efficacy, &tr.anxiety, &tr.readiness, &tr.growth,
                  &tr.engaged, &tr.zpdShare})
    v->assign(sessions, 0.0);
  for (int s = 0; s < sessions; ++s) tr.day.push_back(c.sessionGap * s);
  r.learners.resize(n);

  const double reinforcement = d[Lever::Reinforcement], direction = d[Lever::TeacherDirection], empathy = d[Lever::Empathy],
               choice = d[Lever::Choice], relevance = d[Lever::Relevance], transparency = d[Lever::Transparency],
               prior = d[Lever::PriorActivation], retrieval = d[Lever::RetrievalPractice], spacing = d[Lever::Spacing],
               interleaving = d[Lever::Interleaving], feedback = d[Lever::Feedback], problemFirst = d[Lever::ProblemFirst],
               scaffolding = d[Lever::Scaffolding], dialogue = d[Lever::PeerDialogue], inclusivity = d[Lever::Inclusivity],
               growthFeedback = d[Lever::GrowthFeedback];
  const ZpdParams zpd;

  for (int i = 0; i < n; ++i) {
    Rng rng = Rng::stream(seed, static_cast<std::uint64_t>(i));
    State st;
    st.l = sampleLearner(population_, rng);
    st.units.resize(sessions);
    const double start = st.l.ability;
    const double adult = adultness(st.l.age);
    const auto stage = stageDistribution(st.l.age);
    const double formal = stage[static_cast<int>(PiagetStage::FormalOperational)];
    const double abstractFit = 1.0 - c.abstraction * (1.0 - formal) * (1.0 - 0.5 * scaffolding);
    // Bourdieu: the classroom's culture taxes learners whose home culture differs from it.
    const double access = 1.0 - 0.5 * (1.0 - st.l.capital) * (1.0 - inclusivity);

    std::size_t nextDay = 0;
    double theta = st.l.ability;
    auto report = [&](std::size_t k) {
      const double t = days[k];
      double rec = 0.0;
      for (const auto& u : st.units)
        if (u.introduced) rec += u.learned * u.trace.retrievability(t);
      rec /= sessions;
      const double skill = logistic(theta - finalLevel);
      recall[k][i] = rec;
      score[k][i] = rec * choiceAccuracy(st.discrimination) * (0.5 + 0.5 * skill);
    };

    for (int s = 0; s < sessions; ++s) {
      const double t = c.sessionGap * s;
      while (nextDay < nd && days[nextDay] < t - 1e-9) report(nextDay++);
      std::array<double, kDrawsPerSession> u{};
      for (auto& x : u) x = rng.uniform();
      const double noise = rng.normal();

      // 1. Needs: the classroom climate lifts safety, belonging and esteem.
      NeedState needs;
      needs.satisfaction = {st.l.physiological, clamp01(0.45 + 0.25 * st.l.capital - 0.2 * st.l.anxiety),
                            clamp01(0.35 + 0.35 * st.l.capital), clamp01(0.3 + 0.4 * st.l.efficacy), 0.3};
      needs = applyClassroomClimate(needs, 0.8 * empathy, clamp01(0.5 * inclusivity + 0.3 * dialogue + 0.3 * empathy),
                                    clamp01(0.4 * growthFeedback + 0.3 * feedback));
      const double readiness = learningReadiness(needs);

      // 2. Affect: productive struggle is uncomfortable without a supportive climate.
      st.l.anxiety = clamp01(st.l.anxiety + 0.08 * problemFirst * (1.0 - st.l.growth) * (1.0 - empathy));
      st.l.anxiety *= 1.0 - 0.25 * empathy;
      const double clarity = 1.0 / (1.0 + 0.5 * st.l.anxiety);

      // 3. Motivation: value × expectancy, with andragogy's assumptions weighted by age.
      st.l.intrinsic = std::max(0.0, st.l.intrinsic - 0.01 * reinforcement * st.l.intrinsic);  // overjustification
      const double value = clamp(0.2 + 0.35 * st.l.intrinsic + relevance * (0.15 + 0.2 * adult) +
                                     transparency * (0.05 + 0.15 * adult) + choice * (0.05 + 0.15 * adult) +
                                     reinforcement * (0.05 + 0.3 * (1.0 - adult)) - 0.1 * adult * direction,
                                 0.0, 1.2);
      const double motivation = value * st.l.efficacy;
      if (st.engaged) st.habit += 0.1 * reinforcement * (1.0 - st.habit);
      const double effort = st.engaged ? clamp01(0.35 + 1.0 * motivation + 0.15 * st.habit + st.persistence) : 0.1;

      // 4. Challenge and support (Vygotsky): the task sits at the curriculum level or, with
      //    scaffolding, at the learner's ZPD; teacher and peers act as more knowledgeable others.
      const double level = c.difficulty + c.difficultyStep * s;
      const double b = (1.0 - scaffolding) * level + scaffolding * (theta + 0.75);
      const double support = 1.5 * scaffolding + 0.4 * direction + 0.8 * dialogue * logistic(-theta + 0.5);
      const double pAlone = successAlone(theta, b), pAssisted = successAssisted(theta, b, support);
      ZpdParams here = zpd;
      here.scaffold = support;
      const bool inZpd = zoneOf(theta, b, here) == Zone::Proximal;

      // 5–7. Learning: ability grows in the ZPD; productive struggle helps when it is supported.
      const double struggle = 1.0 + problemFirst * (0.5 * (st.l.growth + empathy) - 0.35);
      const double drive = effort * clarity * (0.6 + 0.4 * readiness) * access * abstractFit;
      theta += 0.9 * drive * pAssisted * (1.0 - pAlone) * (1.0 + 0.3 * dialogue) * (1.0 + 0.2 * feedback) * struggle *
               std::max(0.0, 1.0 + 0.15 * noise);

      // 6. Encoding of the session's unit and practice in the review slots.
      Unit& unit = st.units[s];
      unit.introduced = true;
      unit.learned = clamp01(2.2 * drive * (0.5 + 0.5 * pAssisted) * (0.75 + 0.25 * direction) * (1.0 + 0.1 * transparency) *
                             (1.0 + 0.15 * feedback));
      const double depth = 1.0 + 0.6 * prior + memory_.params().pretestBoost * problemFirst + 0.4 * relevance + 0.3 * dialogue;
      unit.trace = memory_.encode(t, depth);
      for (int k = 0; k < kReviewSlots; ++k) {
        const double uTarget = u[3 + 4 * k], uMode = u[4 + 4 * k], uRecall = u[5 + 4 * k], uFeedback = u[6 + 4 * k];
        // Spaced: an earlier unit chosen at random; massed: the unit just taught.
        const int target = s > 0 && uTarget < spacing ? std::min(s - 1, static_cast<int>(uTarget / spacing * s)) : s;
        Unit& rev = st.units[target];
        const StudyMode mode = uMode < retrieval ? (uFeedback < feedback ? StudyMode::RetrievalWithFeedback : StudyMode::Retrieval)
                                                 : StudyMode::Restudy;
        const bool ok = memory_.study(rev.trace, t + 0.01 * (k + 1), mode, uRecall);
        // Corrective feedback after a failed attempt also fills gaps in what was learned.
        if (!ok && mode == StudyMode::RetrievalWithFeedback) rev.learned = clamp01(rev.learned + 0.15 * (1.0 - rev.learned));
        if (effort > 0.3) rev.learned = clamp01(rev.learned + 0.03 * (1.0 - rev.learned));
      }
      st.discrimination += 0.03 + 0.25 * interleaving;

      // 8. After the session: efficacy, anxiety, mindset and engagement respond to the outcome.
      const bool success = u[0] < pAssisted;
      st.l.efficacy = clamp01(st.l.efficacy + (0.1 + 0.1 * feedback) * ((success ? 1.0 : 0.0) - st.l.efficacy));
      if (!success) {
        // Dweck: a growth mindset meets failure with more effort, a fixed one withdraws it.
        st.persistence = clamp(st.persistence + 0.06 * st.l.growth - 0.08 * (1.0 - st.l.growth), -0.3, 0.2);
        st.l.efficacy = clamp01(st.l.efficacy - 0.04 * (1.0 - st.l.growth) * (1.0 - growthFeedback));
        st.l.anxiety = clamp01(st.l.anxiety + 0.05 * (1.0 - st.l.growth) * (1.0 - empathy));
      } else {
        st.l.anxiety = clamp01(st.l.anxiety - 0.03);
      }
      st.l.growth = clamp01(st.l.growth + 0.05 * growthFeedback * (1.0 - st.l.growth));
      if (st.engaged && !success && effort < 0.45 && st.l.growth < 0.45 && u[1] < 0.25 * (1.0 - empathy)) st.engaged = false;
      else if (!st.engaged && u[2] < 0.3 * empathy + 0.2 * relevance) st.engaged = true;

      tr.ability[s] += (theta - start) / n;
      tr.effort[s] += effort / n;
      tr.motivation[s] += motivation / n;
      tr.value[s] += value / n;
      tr.efficacy[s] += st.l.efficacy / n;
      tr.anxiety[s] += st.l.anxiety / n;
      tr.readiness[s] += readiness / n;
      tr.growth[s] += st.l.growth / n;
      tr.engaged[s] += (st.engaged ? 1.0 : 0.0) / n;
      tr.zpdShare[s] += (inZpd ? 1.0 : 0.0) / n;
    }
    while (nextDay < nd) report(nextDay++);

    LearnerOutcome& o = r.learners[i];
    o.capital = st.l.capital;
    o.abilityGain = theta - start;
    o.engagedAtEnd = st.engaged;
    for (std::size_t k = 0; k < nd; ++k) {
      if (std::fabs(days[k] - endDay) < 1e-9) o.endScore = score[k][i];
      if (std::fabs(days[k] - testDay) < 1e-9) o.delayedScore = score[k][i];
    }
  }

  auto fill = [&](LearningProfile& p, const std::vector<std::vector<double>>& m) {
    p.days = days;
    for (const auto& v : m) {
      const Summary s = summarise(v);
      p.mean.push_back(s.mean);
      p.q10.push_back(s.q10);
      p.q50.push_back(s.median);
      p.q90.push_back(s.q90);
      double risk = 0.0;
      for (double x : v) risk += x < c.atRiskBelow ? 1.0 : 0.0;
      p.atRisk.push_back(risk / static_cast<double>(v.size()));
    }
  };
  fill(r.score, score);
  fill(r.recall, recall);

  std::vector<double> capital(n), delayed(n);
  for (int i = 0; i < n; ++i) {
    capital[i] = r.learners[i].capital;
    delayed[i] = r.learners[i].delayedScore;
    r.endScore += r.learners[i].endScore / n;
    r.delayedScore += delayed[i] / n;
    r.abilityGain += r.learners[i].abilityGain / n;
    r.atRiskDelayed += (delayed[i] < c.atRiskBelow ? 1.0 : 0.0) / n;
  }
  const double median = quantile(capital, 0.5);
  double hi = 0.0, lo = 0.0;
  int nHi = 0, nLo = 0;
  for (int i = 0; i < n; ++i) {
    if (capital[i] > median) {
      hi += delayed[i];
      ++nHi;
    } else {
      lo += delayed[i];
      ++nLo;
    }
  }
  r.equityGap = (nHi ? hi / nHi : 0.0) - (nLo ? lo / nLo : 0.0);
  r.retention = r.endScore > 0.0 ? r.delayedScore / r.endScore : 0.0;
  return r;
}

}  // namespace syn
