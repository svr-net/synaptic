// The classroom: every theory of the chapter acting on the same learners, in the space.
//
// One agent is one learner taking a course of sessions and a delayed test. Every state the
// theories care about is a point: ability a level of the cognitive channel; anxiety, mindset,
// self-efficacy, interest, habit and needs shares of the affective channel; cultural capital and
// belonging shares of the social channel; every learned unit a share and every memory trace a
// point turned away from the learner by forgetting. Each session applies, in order:
//   1. needs (Maslow): safety and belonging, raised by the climate (mixtures), gate readiness —
//      each gate is the equator dilated by (met − threshold)/width, read as a chance;
//   2. affect: problem-first struggle without growth or empathy raises anxiety, empathy lowers it;
//   3. motivation: value from interest, relevance, choice, transparency and rewards, times
//      self-efficacy, gives effort;
//   4. challenge (Vygotsky): the task is the point of logit θ − b, the teacher's and peers'
//      support the dilation shift(T, s); ability is dilated by η·effort·P_assisted·(1 − P_alone);
//   5. development (Piaget): abstract content needs formal operations (a chance of the age);
//   6. memory: the unit is encoded and practised in three review slots (spaced or massed,
//      retrieval or restudy, with or without feedback): retrieval succeeds when the trace lies in
//      the cap of a uniform draw;
//   7. culture (Bourdieu): the capital mismatch taxes access unless teaching is inclusive;
//   8. after the session, efficacy follows the outcome by a mixture, anxiety and mindset respond,
//      and fixed-minded learners who fail with low effort may disengage.
// The test score at day t is what is still retrievable × the chance of telling problems apart
// (a point raised by interleaving) × skill at the final level of the curriculum.
//
// Models sharing this agent: classroom (one design, profiles, quantiles and learner values),
// theories (the eight theory designs on three populations), allocation (Shapley, incremental and
// standalone shares of a design's gain), sensitivity (lever bumps with common random numbers and
// independent seeds) and gagne (Gagné's nine events, each left out in turn).
#include <algorithm>
#include <array>
#include <cmath>
#include <map>
#include <numeric>
#include <string>

#include "syn/models/models.hpp"

// The agent program, in WGSL.
constexpr const char* classroom_agent = R"wgsl(
  let nS = u32(P(36u));
  let nD = u32(P(37u));
  let endIdx = u32(P(38u));
  let testIdx = u32(P(39u));
  let profiles = P(40u) > f32(0.5);
  let sessionOff = u32(P(41u));
  let jobOff = u32(P(42u));
  let keep = P(43u) > f32(0.5);
  let reinforcement = J(0u);
  let direction = J(1u);
  let empathy = J(2u);
  let choice = J(3u);
  let relevance = J(4u);
  let transparency = J(5u);
  let prior = J(6u);
  let retrieval = J(7u);
  let spacing = J(8u);
  let interleaving = J(9u);
  let feedback = J(10u);
  let problemFirst = J(11u);
  let scaffolding = J(12u);
  let dialogue = J(13u);
  let inclusivity = J(14u);
  let growthFeedback = J(15u);

  let age = max(f32(4), P(0u) + P(1u) * kLogisticUnit * logisticOf(U(0u)));
  let capital = sampleShare(U(1u), P(12u), P(13u), kC);
  var ability = shift(sampleLevel(U(2u), P(2u), P(3u), kA), P(14u) * (chance(capital) - P(12u)));
  var anxiety = sampleShare(U(3u), P(4u), P(5u), kB);
  var growth = sampleShare(U(4u), P(6u), P(7u), kB);
  var efficacy = sampleShare(U(5u), P(8u), P(9u), kB);
  var intrinsic = sampleShare(U(6u), P(10u), P(11u), kB);
  let physiological = sampleShare(U(7u), f32(0.85), f32(0.1), kB);
  let start = logit(ability);
  let adult = clamp((age - f32(12)) / f32(10), f32(0), f32(1));
  let formal = chance(atLogit((age - P(35u)) / (P(34u) * kLogisticUnit), kA));
  let abstractFit = f32(1) - P(18u) * (f32(1) - formal) * (f32(1) - f32(0.5) * scaffolding);
  let access = f32(1) - f32(0.5) * (f32(1) - chance(capital)) * (f32(1) - inclusivity);
  var habit = atChance(f32(0), kB);
  var persistence = atChance(f32(0.6), kB);
  var discrimination = atChance(f32(1) - exp(f32(-0.6)), kA);
  var engaged = true;
  var learned : array<V5, 16>;
  var stab : array<f32, 16>;
  var last : array<f32, 16>;
  var units = 0u;
  var nextDay = 0u;
  var endScore = f32(0);
  var delayed = f32(0);

  for (var s = 0u; s <= nS; s = s + 1u) {
    let now = select(P(15u) * f32(s), f32(1e30), s == nS);
    while (nextDay < nD && P(48u + nextDay) < now - f32(1e-4)) {
      let day = P(48u + nextDay);
      var recalled = f32(0);
      for (var k = 0u; k < units; k = k + 1u) {
        let r = select(f32(1), f32(1) / sqrt(f32(1) + f32(19) / f32(81) * (day - last[k]) / stab[k]), day > last[k]);
        recalled = recalled + chance(learned[k]) * chance(rotate(pole(), kN, kA, f32(2) * acos(sqrt(r))));
      }
      recalled = recalled / f32(nS);
      let skill = chance(shift(ability, -P(20u)));
      let telling = f32(1) / f32(3) + f32(2) / f32(3) * chance(discrimination);
      let score = recalled * telling * (f32(0.5) + f32(0.5) * skill);
      stat(nextDay * 4u, score);
      stat(nextDay * 4u + 1u, score * score);
      stat(nextDay * 4u + 2u, recalled);
      stat(nextDay * 4u + 3u, select(f32(0), f32(1), score < P(19u)));
      if (keep) { quant(nextDay, score); }
      if (nextDay == endIdx) { endScore = score; }
      if (nextDay == testIdx) { delayed = score; }
      nextDay = nextDay + 1u;
    }
    if (s == nS) { break; }

    let k0 = 16u + s * 24u;
    let safety0 = clamp(f32(0.45) + f32(0.25) * chance(capital) - f32(0.2) * chance(anxiety), f32(0), f32(1));
    let belonging0 = clamp(f32(0.35) + f32(0.35) * chance(capital), f32(0), f32(1));
    let safety = raise(atChance(safety0, kB), clamp(f32(0.8) * empathy, f32(0), f32(1)));
    let belonging = raise(atChance(belonging0, kC), clamp(f32(0.5) * inclusivity + f32(0.3) * dialogue + f32(0.3) * empathy, f32(0), f32(1)));
    let readiness = chance(shift(equator(kB), (chance(physiological) - P(29u)) / P(30u))) *
                    chance(shift(equator(kB), (chance(safety) - P(29u)) / P(30u))) *
                    chance(shift(equator(kB), (chance(belonging) - P(29u)) / P(30u)));

    anxiety = nudge(anxiety, f32(0.08) * problemFirst * (f32(1) - chance(growth)) * (f32(1) - empathy));
    anxiety = lower(anxiety, f32(0.25) * empathy);
    let clarity = f32(1) / (f32(1) + f32(0.5) * chance(anxiety));

    intrinsic = lower(intrinsic, f32(0.01) * reinforcement);
    let value = clamp(f32(0.2) + f32(0.35) * chance(intrinsic) + relevance * (f32(0.15) + f32(0.2) * adult) +
                      transparency * (f32(0.05) + f32(0.15) * adult) + choice * (f32(0.05) + f32(0.15) * adult) +
                      reinforcement * (f32(0.05) + f32(0.3) * (f32(1) - adult)) - f32(0.1) * adult * direction, f32(0), f32(1.2));
    let motivation = value * chance(efficacy);
    if (engaged) { habit = raise(habit, f32(0.1) * reinforcement); }
    let effort = select(f32(0.1), clamp(f32(0.35) + motivation + f32(0.15) * chance(habit) + f32(0.5) * chance(persistence) - f32(0.3), f32(0), f32(1)), engaged);

    let theta = logit(ability);
    let level = P(16u) + P(17u) * f32(s);
    let b = (f32(1) - scaffolding) * level + scaffolding * (theta + f32(0.75));
    let support = f32(1.5) * scaffolding + f32(0.4) * direction + f32(0.8) * dialogue * chance(atLogit(f32(0.5) - theta, kC));
    let task = atLogit(theta - b, kA);
    let helped = shift(task, support);
    let pAlone = chance(task);
    let pAssisted = chance(helped);
    let inZpd = !inside(task, chanceCap(P(31u))) && inside(helped, chanceCap(P(32u)));

    let struggle = f32(1) + problemFirst * (f32(0.5) * (chance(growth) + empathy) - f32(0.35));
    let drive = effort * clarity * (f32(0.6) + f32(0.4) * readiness) * access * abstractFit;
    let noise = kLogisticUnit * logisticOf(U(k0 + 16u));
    ability = shift(ability, f32(0.9) * drive * pAssisted * (f32(1) - pAlone) * (f32(1) + f32(0.3) * dialogue) * (f32(1) + f32(0.2) * feedback) *
                                 struggle * max(f32(0), f32(1) + f32(0.15) * noise));

    learned[s] = atChance(clamp(f32(2.2) * drive * (f32(0.5) + f32(0.5) * pAssisted) * (f32(0.75) + f32(0.25) * direction) *
                                    (f32(1) + f32(0.1) * transparency) * (f32(1) + f32(0.15) * feedback), f32(0), f32(1)), kA);
    stab[s] = P(21u) * (f32(1) + f32(0.6) * prior + P(28u) * problemFirst + f32(0.4) * relevance + f32(0.3) * dialogue);
    last[s] = P(15u) * f32(s);
    units = s + 1u;
    for (var k = 0u; k < 3u; k = k + 1u) {
      let uTarget = U(k0 + 3u + 4u * k);
      let uMode = U(k0 + 4u + 4u * k);
      let uRecall = U(k0 + 5u + 4u * k);
      let uFeedback = U(k0 + 6u + 4u * k);
      var unit = s;
      if (s > 0u && uTarget < spacing) { unit = min(s - 1u, u32(uTarget / spacing * f32(s))); }
      let at = P(15u) * f32(s) + f32(0.01) * f32(k + 1u);
      let r = select(f32(1), f32(1) / sqrt(f32(1) + f32(19) / f32(81) * (at - last[unit]) / stab[unit]), at > last[unit]);
      let trace = rotate(pole(), kN, kA, f32(2) * acos(sqrt(r)));
      if (uMode < retrieval) {
        if (inside(trace, chanceCap(uRecall))) {
          stab[unit] = stab[unit] * (f32(1) + P(24u) + P(25u) * (f32(1) - r));
          last[unit] = at;
        } else if (uFeedback < feedback) {
          stab[unit] = max(P(27u), P(26u) * stab[unit]) * (f32(1) + P(22u) + P(23u) * (f32(1) - r));
          last[unit] = at;
          learned[unit] = raise(learned[unit], f32(0.15));
        }
      } else {
        stab[unit] = stab[unit] * (f32(1) + P(22u) + P(23u) * (f32(1) - r));
        last[unit] = at;
      }
      if (effort > f32(0.3)) { learned[unit] = raise(learned[unit], f32(0.03)); }
    }
    discrimination = raise(discrimination, f32(1) - exp(-(f32(0.03) + f32(0.25) * interleaving)));

    let succeeded = inside(helped, chanceCap(U(k0)));
    efficacy = blend(efficacy, atChance(select(f32(0), f32(1), succeeded), kB), f32(0.1) + f32(0.1) * feedback);
    if (!succeeded) {
      let g = chance(growth);
      persistence = nudge(persistence, (f32(0.06) * g - f32(0.08) * (f32(1) - g)) / f32(0.5));
      efficacy = nudge(efficacy, -f32(0.04) * (f32(1) - g) * (f32(1) - growthFeedback));
      anxiety = nudge(anxiety, f32(0.05) * (f32(1) - g) * (f32(1) - empathy));
    } else {
      anxiety = nudge(anxiety, f32(-0.03));
    }
    growth = raise(growth, f32(0.05) * growthFeedback);
    if (engaged && !succeeded && effort < f32(0.45) && chance(growth) < f32(0.45) && U(k0 + 1u) < f32(0.25) * (f32(1) - empathy)) {
      engaged = false;
    } else if (!engaged && U(k0 + 2u) < f32(0.3) * empathy + f32(0.2) * relevance) {
      engaged = true;
    }

    if (profiles) {
      let o = sessionOff + s * 10u;
      stat(o, logit(ability) - start);
      stat(o + 1u, effort);
      stat(o + 2u, motivation);
      stat(o + 3u, value);
      stat(o + 4u, chance(efficacy));
      stat(o + 5u, chance(anxiety));
      stat(o + 6u, readiness);
      stat(o + 7u, chance(growth));
      stat(o + 8u, f32(engaged));
      stat(o + 9u, f32(inZpd));
    }
  }
  let high = chance(capital) > P(33u);
  stat(jobOff, endScore);
  stat(jobOff + 1u, delayed);
  stat(jobOff + 2u, logit(ability) - start);
  stat(jobOff + 3u, delayed * f32(high));
  stat(jobOff + 4u, f32(high));
  if (keep) {
    quant(nD, chance(capital));
    quant(nD + 1u, delayed);
    quant(nD + 2u, chance(atLogit(f32(0.5) * (logit(ability) - start), kA)));
  }
)wgsl";

namespace syn::models {

namespace {

constexpr int kLevers = 16;
constexpr std::uint32_t kMaxStats = 680;
constexpr int kMaxDays = 128;
using Design = std::array<double, kLevers>;

const char* kLeverKeys[kLevers] = {"reinforcement", "teacherDirection", "empathy", "choice", "relevance", "transparency", "priorActivation",
                                   "retrievalPractice", "spacing", "interleaving", "feedback", "problemFirst", "scaffolding", "peerDialogue",
                                   "inclusivity", "growthFeedback"};
const char* kLeverNames[kLevers] = {"reinforcement", "teacher direction", "empathy", "choice", "relevance", "transparency",
                                    "prior activation", "retrieval practice", "spacing", "interleaving", "feedback", "problem first",
                                    "scaffolding (ZPD)", "peer dialogue", "inclusivity", "growth feedback"};
const char* kLeverTheories[kLevers] = {"behaviourism", "behaviourism / cognitivism", "humanism", "humanism / andragogy", "andragogy / motivation",
                                       "Gagné / andragogy", "constructivism / Gagné", "cognitivism", "cognitivism", "cognitivism",
                                       "Gagné / motivation", "constructivism", "Vygotsky (ZPD)", "social constructivism",
                                       "Bourdieu / critical practice", "growth mindset"};
const char* kTheories[8] = {"banking lecture", "behaviourism", "humanism", "cognitivism", "constructivism", "social constructivism", "andragogy", "blended"};

// The lesson each theory would design (Activity 3.3 in lever form).
//                                            rein dir  emp  cho  rel  tra  pri  ret  spa  int  fb   pf   sca  dia  inc  gro
const Design kTheoryDesigns[8] = {{0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                                  {1, 1, 0, 0, 0, 0, 0, 0.3, 0, 0, 0.6, 0, 0, 0, 0, 0},
                                  {0, 0.2, 1, 0.8, 0.9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.5, 0.3},
                                  {0, 0.8, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0.4, 0.3, 0, 0, 0},
                                  {0, 0.2, 0.6, 0.4, 0.6, 0, 1, 0, 0, 0, 0, 1, 0.4, 0, 0, 0},
                                  {0, 0.3, 0.5, 0, 0, 0, 0.5, 0, 0, 0, 0, 0.4, 1, 1, 1, 0},
                                  {0, 0.2, 0.5, 1, 1, 1, 0.8, 0, 0, 0, 0, 0.5, 0, 0, 0, 0},
                                  {0.3, 0.5, 0.8, 0.5, 0.8, 1, 0.8, 0.9, 0.9, 0.7, 1, 0.5, 0.8, 0.7, 0.8, 0.8}};

const char* kGagneEvents[9] = {"gain attention", "inform the learner of the objective", "stimulate recall of prior learning", "present the stimulus",
                               "provide learner guidance", "elicit performance", "provide feedback", "assess performance",
                               "enhance retention and transfer"};

Design gagneDesign(const std::array<double, 9>& x) {
  Design d{};
  d[4] = 0.5 * x[0];
  d[5] = x[1];
  d[6] = x[2];
  d[1] = x[3];
  d[12] = std::min(1.0, 0.6 * x[4] + 0.3 * x[7]);
  d[7] = std::min(1.0, 0.8 * x[5] + 0.2 * x[7]);
  d[10] = x[6];
  d[8] = x[8];
  d[9] = 0.6 * x[8];
  return d;
}

Value designValue(const Design& d) {
  Value v = Value::object();
  for (int k = 0; k < kLevers; ++k) v.set(kLeverKeys[k], d[k]);
  return v;
}

// A specification without a design gets the blended one, as on the web pages.
Design designOf(const Value& v) {
  if (!v.isObject() || v.asObject().empty()) return kTheoryDesigns[7];
  Design d{};
  for (int k = 0; k < kLevers; ++k) d[k] = std::min(1.0, std::max(0.0, v.num(kLeverKeys[k], 0.0)));
  return d;
}

struct Population {
  std::string name;
  double age = 16, ageSd = 0.5, priorMean = 0, priorSd = 1, anxietyMean = 0.4, anxietySd = 0.2, growthMean = 0.5, growthSd = 0.25,
         efficacyMean = 0.55, efficacySd = 0.15, intrinsicMean = 0.35, intrinsicSd = 0.2, capitalMean = 0.55, capitalSd = 0.25, capitalPrior = 0.5;
};

Population preset(const std::string& name) {
  Population p;
  p.name = "secondary school (16)";
  if (name == "children") {
    p.name = "primary school (9)";
    p.age = 9;
    p.anxietyMean = 0.3;
    p.growthMean = 0.6;
    p.efficacyMean = 0.6;
    p.intrinsicMean = 0.4;
  } else if (name == "adults") {
    p.name = "adult learners (35)";
    p.age = 35;
    p.ageSd = 8;
    p.anxietyMean = 0.45;
    p.growthMean = 0.45;
    p.efficacyMean = 0.5;
    p.intrinsicMean = 0.55;
  }
  return p;
}

Population populationOf(const Value& v) {
  Population p = preset(v.str("preset", "secondary"));
  p.age = v.num("age", p.age);
  p.priorMean = v.num("priorMean", p.priorMean);
  p.priorSd = v.num("priorSd", p.priorSd);
  p.anxietyMean = v.num("anxietyMean", p.anxietyMean);
  p.growthMean = v.num("growthMean", p.growthMean);
  p.efficacyMean = v.num("efficacyMean", p.efficacyMean);
  p.intrinsicMean = v.num("intrinsicMean", p.intrinsicMean);
  p.capitalMean = v.num("capitalMean", p.capitalMean);
  p.capitalSd = v.num("capitalSd", p.capitalSd);
  return p;
}

struct Course {
  int sessions = 8;
  double gap = 7, difficulty = 0.3, step = 0.1, delayedTest = 30, abstraction = 0.6, atRisk = 0.3;
};

Course courseOf(const Value& v) {
  Course c;
  c.sessions = v.integer("sessions", c.sessions);
  c.gap = v.num("sessionGap", c.gap);
  c.difficulty = v.num("difficulty", c.difficulty);
  c.step = v.num("difficultyStep", c.step);
  c.delayedTest = v.num("delayedTest", c.delayedTest);
  c.abstraction = v.num("abstraction", c.abstraction);
  c.atRisk = v.num("atRiskBelow", c.atRisk);
  if (c.sessions < 1 || c.sessions > 16) throw std::invalid_argument("classroom: a course has 1 to 16 sessions (a learner keeps 16 units)");
  if (c.gap <= 0 || c.delayedTest < 0) throw std::invalid_argument("classroom: need sessionGap > 0 and delayedTest >= 0");
  return c;
}

struct Layout {
  std::vector<double> days;
  std::uint32_t endIdx = 0, testIdx = 0, sessionOff = 0, jobOff = 0, stats = 0;
  double lastSession = 0, testDay = 0;
};

Layout layoutOf(const Course& c, bool profiles) {
  Layout l;
  l.lastSession = c.gap * (c.sessions - 1);
  l.testDay = l.lastSession + c.delayedTest;
  const double endDay = l.lastSession + 1.0;
  if (profiles) {
    double step = 1.0;
    while ((l.testDay / step) + 3 > kMaxDays) step += 1.0;
    for (double t = 0.0; t <= l.testDay + 1e-9; t += step) l.days.push_back(t);
    l.days.push_back(endDay);
    l.days.push_back(l.testDay);
    std::sort(l.days.begin(), l.days.end());
    l.days.erase(std::unique(l.days.begin(), l.days.end(), [](double a, double b) { return std::fabs(a - b) < 1e-9; }), l.days.end());
  } else {
    l.days = {endDay, l.testDay};
  }
  for (std::size_t k = 0; k < l.days.size(); ++k) {
    if (std::fabs(l.days[k] - endDay) < 1e-9) l.endIdx = static_cast<std::uint32_t>(k);
    if (std::fabs(l.days[k] - l.testDay) < 1e-9) l.testIdx = static_cast<std::uint32_t>(k);
  }
  l.sessionOff = static_cast<std::uint32_t>(l.days.size() * 4);
  l.jobOff = l.sessionOff + (profiles ? static_cast<std::uint32_t>(c.sessions) * 10 : 0);
  l.stats = l.jobOff + 5;
  return l;
}

struct Job {
  Design design;
  std::uint32_t seed;
};

Plan compileClassroom(const std::string& model, const Population& pop, const Course& c, std::uint32_t learners, const std::vector<Job>& jobs,
                      bool profiles, bool keep) {
  const Layout l = layoutOf(c, profiles);
  std::vector<double> params = {pop.age, pop.ageSd, pop.priorMean, pop.priorSd, pop.anxietyMean, pop.anxietySd, pop.growthMean, pop.growthSd,
                                pop.efficacyMean, pop.efficacySd, pop.intrinsicMean, pop.intrinsicSd, pop.capitalMean, pop.capitalSd,
                                pop.capitalPrior, c.gap, c.difficulty, c.step, c.abstraction, c.atRisk, c.difficulty + c.step * (c.sessions - 1),
                                // memory: initial stability, restudy floor/gain, retrieval floor/gain, failure, min stability, pretest
                                1.0, 0.05, 0.6, 0.30, 5.0, 0.7, 0.5, 0.5,
                                // Maslow threshold and width; ZPD thresholds; capital split; Piaget onset sd and formal onset
                                0.55, 0.07, 0.8, 0.5, std::min(1.0, std::max(0.0, pop.capitalMean)), 0.9, 12.0,
                                double(c.sessions), double(l.days.size()), double(l.endIdx), double(l.testIdx), profiles ? 1.0 : 0.0,
                                double(l.sessionOff), double(l.jobOff), keep ? 1.0 : 0.0, 0, 0, 0, 0};
  params.insert(params.end(), l.days.begin(), l.days.end());
  std::vector<std::vector<double>> rows;
  for (const auto& j : jobs) rows.push_back(std::vector<double>(j.design.begin(), j.design.end()));
  const std::uint32_t q = keep ? static_cast<std::uint32_t>(l.days.size() + 3) : 0;
  Plan p = makePlan(model, learners, l.stats, params, rows, 0, q);
  for (std::size_t k = 0; k < jobs.size(); ++k) p.seeds[k] = jobs[k].seed;
  if (keep) p.keepAgents = std::min<std::uint32_t>(learners, 400);
  return p;
}

// What a classroom job's means say: profiles, session traces and the summary.
Value summary(const Plan& p, const Output& o, std::uint32_t job, const Course& c) {
  const Layout l = layoutOf(c, p.params[40] > 0.5);
  Value r = Value::object();
  const double n = p.agents;
  const double end = o.mean(p, job, l.jobOff), delayed = o.mean(p, job, l.jobOff + 1);
  const double highCount = o.mean(p, job, l.jobOff + 4) * n, lowCount = n - highCount;
  const double highSum = o.mean(p, job, l.jobOff + 3) * n, lowSum = delayed * n - highSum;
  r.set("endScore", end).set("delayedScore", delayed).set("retention", end > 0 ? delayed / end : 0.0);
  r.set("equityGap", (highCount > 0 ? highSum / highCount : 0.0) - (lowCount > 0 ? lowSum / lowCount : 0.0));
  r.set("atRiskDelayed", o.mean(p, job, l.testIdx * 4 + 3)).set("abilityGain", o.mean(p, job, l.jobOff + 2));
  r.set("lastSessionDay", l.lastSession).set("delayedTestDay", l.testDay).set("learners", static_cast<double>(p.agents));
  return r;
}

Value profiles(const Plan& p, const Output& o, std::uint32_t job, const Course& c, Value r) {
  const Layout l = layoutOf(c, true);
  const std::size_t D = l.days.size();
  std::vector<double> mean, atRisk, recall, q10, q50, q90, nan(D, std::nan(""));
  for (std::size_t k = 0; k < D; ++k) {
    mean.push_back(o.mean(p, job, static_cast<std::uint32_t>(k * 4)));
    recall.push_back(o.mean(p, job, static_cast<std::uint32_t>(k * 4 + 2)));
    atRisk.push_back(o.mean(p, job, static_cast<std::uint32_t>(k * 4 + 3)));
    const bool q = p.quantiles > 0;
    q10.push_back(q ? o.quantiles[(job * p.quantiles + k) * 3] : std::nan(""));
    q50.push_back(q ? o.quantiles[(job * p.quantiles + k) * 3 + 1] : std::nan(""));
    q90.push_back(q ? o.quantiles[(job * p.quantiles + k) * 3 + 2] : std::nan(""));
  }
  auto profile = [&](const std::vector<double>& m, const std::vector<double>& a, const std::vector<double>& lo, const std::vector<double>& mid,
                     const std::vector<double>& hi) {
    return Value::object().set("days", Value::Numbers(l.days)).set("mean", Value::Numbers(m)).set("q10", Value::Numbers(lo)).set("q50", Value::Numbers(mid))
        .set("q90", Value::Numbers(hi)).set("atRisk", Value::Numbers(a));
  };
  r.set("score", profile(mean, atRisk, q10, q50, q90));
  r.set("recall", profile(recall, nan, nan, nan, nan));
  Value s = Value::object();
  const char* keys[] = {"ability", "effort", "motivation", "value", "efficacy", "anxiety", "readiness", "growth", "engaged", "zpdShare"};
  std::vector<double> days;
  for (int i = 0; i < c.sessions; ++i) days.push_back(c.gap * i);
  s.set("day", Value::Numbers(days));
  for (int k = 0; k < 10; ++k) {
    std::vector<double> v;
    for (int i = 0; i < c.sessions; ++i) v.push_back(o.mean(p, job, l.sessionOff + i * 10 + k));
    s.set(keys[k], Value::Numbers(v));
  }
  r.set("sessions", s);
  return r;
}

std::vector<double> delayedScores(const Plan& p, const Output& o, const Course& c) {
  const Layout l = layoutOf(c, p.params[40] > 0.5);
  std::vector<double> v;
  for (std::uint32_t j = 0; j < p.jobs; ++j) v.push_back(o.mean(p, j, l.jobOff + 1));
  return v;
}

// ---- allocation: the designs to evaluate and the shares of the gain.
struct Allocation {
  Design design, baseline;
  int permutations = 12;
  std::vector<int> active;
  std::vector<std::uint64_t> masks;
  std::vector<std::vector<int>> orders;
  std::vector<Design> designs;
};

Allocation allocationOf(const Value& spec) {
  const Value& a = spec.sub("allocation");
  Allocation q;
  q.design = designOf(spec.sub("design"));
  q.baseline = a.has("baseline") ? designOf(a.at("baseline")) : kTheoryDesigns[0];
  q.permutations = std::max(1, a.integer("permutations", 12));
  for (int k = 0; k < kLevers; ++k)
    if (std::fabs(q.design[k] - q.baseline[k]) > 1e-12) q.active.push_back(k);
  const std::size_t m = q.active.size();
  const std::uint64_t full = m == 0 ? 0ULL : (~0ULL >> (64 - m));
  std::map<std::uint64_t, bool> seen;
  auto need = [&](std::uint64_t mask) {
    if (seen.emplace(mask, true).second) q.masks.push_back(mask);
  };
  need(0ULL);
  need(full);
  for (int p = 0; p < q.permutations; ++p) {
    std::vector<int> order(m);
    std::iota(order.begin(), order.end(), 0);
    for (int i = static_cast<int>(m) - 1; i > 0; --i)
      std::swap(order[i], order[std::min(i, static_cast<int>(hostUniform(7u, static_cast<std::uint32_t>(p), static_cast<std::uint32_t>(i)) * (i + 1)))]);
    std::uint64_t mask = 0ULL;
    for (int j : order) need(mask |= 1ULL << j);
    q.orders.push_back(order);
  }
  for (std::size_t j = 0; j < m; ++j) {
    need(full & ~(1ULL << j));
    need(1ULL << j);
  }
  for (auto mask : q.masks) {
    Design d = q.baseline;
    for (std::size_t j = 0; j < m; ++j)
      if (mask >> j & 1ULL) d[q.active[j]] = q.design[q.active[j]];
    q.designs.push_back(d);
  }
  return q;
}

// ---- sensitivity: lever bumps with common random numbers and independent seeds.
std::uint32_t streamSeed(std::uint32_t base, std::uint32_t stream) { return hash32(base ^ hash32(stream * 0x9e3779b9u)); }

struct Sensitivity {
  Design design;
  double bump = 0.1;
  int replications = 6;
  std::vector<Job> runs;
};

Sensitivity sensitivityOf(const Value& spec) {
  const Value& s = spec.sub("sensitivity");
  Sensitivity q;
  q.design = designOf(spec.sub("design"));
  q.bump = s.num("bump", 0.1);
  q.replications = std::max(2, s.integer("replications", 6));
  if (!(q.bump > 0.0 && q.bump <= 0.5)) throw std::invalid_argument("sensitivity: need 0 < bump <= 0.5");
  const std::uint32_t base = seedOf(spec);
  q.runs.push_back({q.design, base});
  for (int k = 0; k < kLevers; ++k) {
    Design up = q.design, down = q.design;
    up[k] = std::min(1.0, q.design[k] + q.bump);
    down[k] = std::max(0.0, q.design[k] - q.bump);
    for (int rep = 0; rep < q.replications; ++rep) {
      const std::uint32_t seed = streamSeed(base, 1000u + rep), other = streamSeed(base, 5000u + rep);
      q.runs.push_back({up, seed});
      q.runs.push_back({down, seed});
      q.runs.push_back({down, other});
    }
  }
  return q;
}

std::uint32_t sensitivityLearners(const Value& spec) {
  return static_cast<std::uint32_t>(std::min(spec.sub("sensitivity").integer("maxLearners", 300), static_cast<int>(learnersOf(spec, 500))));
}

kernel::Model base(const char* name, const char* title) {
  kernel::Model m;
  m.name = name;
  m.title = title;
  m.maxStats = kMaxStats;
  m.agentSource = classroom_agent;
  return m;
}

}  // namespace

Value defaults() {
  Value levers = Value::array();
  for (int k = 0; k < kLevers; ++k)
    levers.push(Value::object().set("key", kLeverKeys[k]).set("name", kLeverNames[k]).set("theory", kLeverTheories[k]));
  // in order (a Value object keeps its keys sorted): the lists the pages offer
  Value presets = Value::array();
  for (int t = 0; t < 8; ++t) presets.push(Value::object().set("name", kTheories[t]).set("design", designValue(kTheoryDesigns[t])));
  Value populations = Value::array();
  Value population;
  for (const char* name : {"children", "secondary", "adults"}) {
    const Population p = preset(name);
    populations.push(Value::object().set("preset", name).set("label", p.name).set("age", p.age).set("anxietyMean", p.anxietyMean)
                              .set("growthMean", p.growthMean).set("efficacyMean", p.efficacyMean).set("intrinsicMean", p.intrinsicMean)
                              .set("capitalMean", p.capitalMean));
    if (std::string(name) == "secondary") population = populations.asArray().back();
  }
  const Population s = preset("secondary");
  const Course c;
  population.set("priorMean", s.priorMean).set("priorSd", s.priorSd).set("capitalSd", s.capitalSd);
  // the chapter's Activity 3.3: a short course on keywords and Boolean operators
  const Value spec = Value::object()
                         .set("population", population)
                         .set("course", Value::object().set("sessions", c.sessions).set("sessionGap", c.gap).set("difficulty", c.difficulty)
                                            .set("difficultyStep", c.step).set("delayedTest", c.delayedTest).set("abstraction", c.abstraction)
                                            .set("atRiskBelow", c.atRisk))
                         .set("designPreset", "blended")
                         .set("design", designValue(kTheoryDesigns[7]))
                         .set("sim", Value::object().set("learners", 500).set("seed", 2024));
  return Value::object().set("levers", levers).set("theories", presets).set("populations", populations).set("spec", spec);
}

const kernel::Model* classroomModel() {
  static const kernel::Model model = [] {
    kernel::Model m = base("classroom", "The classroom: every theory acting on the same learners over a course, with a delayed test");
    m.compile = [](const Value& spec) {
      return std::vector<Plan>{compileClassroom("classroom", populationOf(spec.sub("population")), courseOf(spec.sub("course")), learnersOf(spec, 500),
                                                {{designOf(spec.sub("design")), seedOf(spec)}}, true, true)};
    };
    m.summarise = [](const Value& spec, const std::vector<Plan>& plans, const std::vector<Output>& out) {
      const Course c = courseOf(spec.sub("course"));
      const Plan& p = plans[0];
      Value r = profiles(p, out[0], 0, c, summary(p, out[0], 0, c));
      const std::uint32_t keep = std::min(p.keepAgents, p.agents), Q = p.quantiles, D = Q - 3;
      std::vector<double> capital, delayed, gain;
      for (std::uint32_t a = 0; a < keep; ++a) {
        capital.push_back(out[0].agentValues[a * Q + D]);
        delayed.push_back(out[0].agentValues[a * Q + D + 1]);
        const double g = out[0].agentValues[a * Q + D + 2];
        gain.push_back(2.0 * std::log(g / (1.0 - g)));
      }
      r.set("capital", Value::Numbers(capital)).set("delayed", Value::Numbers(delayed)).set("gain", Value::Numbers(gain));
      r.set("delayedHistogram", histogramOf(delayed, 30));
      return r;
    };
    return m;
  }();
  return &model;
}

const kernel::Model* theoriesModel() {
  static const kernel::Model model = [] {
    kernel::Model m = base("theories", "Activity 3.3: the lesson each theory designs, on children, teenagers and adults");
    m.compile = [](const Value& spec) {
      std::vector<Job> jobs;
      for (const auto& d : kTheoryDesigns) jobs.push_back({d, seedOf(spec)});
      std::vector<Plan> plans;
      for (const char* name : {"children", "secondary", "adults"})
        plans.push_back(compileClassroom("theories", preset(name), courseOf(spec.sub("course")), learnersOf(spec, 500), jobs, true, false));
      return plans;
    };
    m.summarise = [](const Value& spec, const std::vector<Plan>& plans, const std::vector<Output>& out) {
      const Course c = courseOf(spec.sub("course"));
      Value r = Value::object();
      r.set("theories", std::vector<std::string>(kTheories, kTheories + 8));
      Value designs = Value::array();
      for (const auto& d : kTheoryDesigns) designs.push(designValue(d));
      r.set("designs", designs);
      Value pops = Value::array();
      const char* names[] = {"children", "secondary", "adults"};
      for (std::size_t i = 0; i < plans.size(); ++i) {
        Value list = Value::array();
        const Layout l = layoutOf(c, true);
        std::vector<double> delayed;
        for (std::uint32_t j = 0; j < plans[i].jobs; ++j) {
          Value x = summary(plans[i], out[i], j, c);
          delayed.push_back(x.num("delayedScore", 0));
          std::vector<double> mean;
          for (std::size_t k = 0; k < l.days.size(); ++k) mean.push_back(out[i].mean(plans[i], j, static_cast<std::uint32_t>(k * 4)));
          x.set("days", Value::Numbers(l.days)).set("mean", Value::Numbers(mean));
          list.push(x);
        }
        // the best design, and the best of the single theories (all but the banking lecture and the blend)
        const std::vector<double> single(delayed.begin() + 1, delayed.end() - 1);
        pops.push(Value::object().set("population", preset(names[i]).name).set("results", list).set("best", bestOf(delayed))
                      .set("bestSingle", 1 + bestOf(single)));
      }
      r.set("populations", pops);
      return r;
    };
    return m;
  }();
  return &model;
}

const kernel::Model* allocationModel() {
  static const kernel::Model model = [] {
    kernel::Model m = base("allocation", "Gain allocation: Shapley, incremental and standalone shares of a design's learning gain");
    m.compile = [](const Value& spec) {
      std::vector<Job> jobs;
      for (const auto& d : allocationOf(spec).designs) jobs.push_back({d, seedOf(spec)});
      return std::vector<Plan>{compileClassroom("allocation", populationOf(spec.sub("population")), courseOf(spec.sub("course")), learnersOf(spec, 500), jobs,
                                                false, false)};
    };
    m.summarise = [](const Value& spec, const std::vector<Plan>& plans, const std::vector<Output>& out) {
      const Allocation q = allocationOf(spec);
      const auto scores = delayedScores(plans[0], out[0], courseOf(spec.sub("course")));
      std::map<std::uint64_t, double> f;
      for (std::size_t i = 0; i < q.masks.size(); ++i) f[q.masks[i]] = scores[i];
      const std::size_t mcount = q.active.size();
      const std::uint64_t full = mcount == 0 ? 0ULL : (~0ULL >> (64 - mcount));
      const double baseScore = f.at(0ULL), designScore = f.at(full);
      std::vector<double> shapley(mcount, 0.0);
      for (const auto& order : q.orders) {
        std::uint64_t mask = 0ULL;
        double prev = baseScore;
        for (int j : order) {
          mask |= 1ULL << j;
          shapley[j] += (f.at(mask) - prev) / q.permutations;
          prev = f.at(mask);
        }
      }
      Value levers = Value::array();
      double sum = 0.0;
      double standaloneSum = 0, incrementalSum = 0;
      for (std::size_t j = 0; j < mcount; ++j) {
        const int k = q.active[j];
        sum += shapley[j];
        standaloneSum += f.at(1ULL << j) - baseScore;
        incrementalSum += designScore - f.at(full & ~(1ULL << j));
        levers.push(Value::object()
                        .set("lever", kLeverNames[k])
                        .set("theory", kLeverTheories[k])
                        .set("baseline", q.baseline[k])
                        .set("design", q.design[k])
                        .set("shapley", shapley[j])
                        .set("incremental", designScore - f.at(full & ~(1ULL << j)))
                        .set("standalone", f.at(1ULL << j) - baseScore));
      }
      return Value::object()
          .set("baselineScore", baseScore)
          .set("designScore", designScore)
          .set("totalGain", designScore - baseScore)
          .set("shapleySum", sum)
          .set("standaloneSum", standaloneSum)
          .set("incrementalSum", incrementalSum)
          .set("permutations", q.permutations)
          .set("evaluations", static_cast<double>(q.masks.size()))
          .set("levers", levers)
          .set("ranked", Value::Numbers(rankedBy(shapley)));
    };
    return m;
  }();
  return &model;
}

const kernel::Model* sensitivityModel() {
  static const kernel::Model model = [] {
    kernel::Model m = base("sensitivity", "Lever sensitivities with common random numbers against independent seeds");
    m.compile = [](const Value& spec) {
      return std::vector<Plan>{compileClassroom("sensitivity", populationOf(spec.sub("population")), courseOf(spec.sub("course")), sensitivityLearners(spec),
                                                sensitivityOf(spec).runs, false, false)};
    };
    m.summarise = [](const Value& spec, const std::vector<Plan>& plans, const std::vector<Output>& out) {
      const Sensitivity q = sensitivityOf(spec);
      const auto scores = delayedScores(plans[0], out[0], courseOf(spec.sub("course")));
      auto index = [&](int k, int rep, int which) { return 1 + (static_cast<std::size_t>(k) * q.replications + rep) * 3 + which; };
      auto meanSe = [](const std::vector<double>& v) {
        double avg = 0;
        for (double x : v) avg += x / v.size();
        double s2 = 0;
        for (double x : v) s2 += (x - avg) * (x - avg) / (v.size() - 1);
        return std::make_pair(avg, std::sqrt(s2 / v.size()));
      };
      Value levers = Value::array();
      std::vector<double> crnMeans;
      double ratio = 0;
      int count = 0;
      for (int k = 0; k < kLevers; ++k) {
        const double v = q.design[k], width = std::min(1.0, v + q.bump) - std::max(0.0, v - q.bump);
        std::vector<double> crn, ind;
        for (int rep = 0; rep < q.replications; ++rep) {
          crn.push_back((scores[index(k, rep, 0)] - scores[index(k, rep, 1)]) / width);
          ind.push_back((scores[index(k, rep, 0)] - scores[index(k, rep, 2)]) / width);
        }
        const auto [cm, cs] = meanSe(crn);
        const auto [im, is] = meanSe(ind);
        crnMeans.push_back(cm);
        if (cs > 0) {
          ratio += (is / cs) * (is / cs);
          ++count;
        }
        levers.push(Value::object().set("lever", kLeverNames[k]).set("theory", kLeverTheories[k]).set("value", v).set("crn", cm).set("crnSe", cs)
                        .set("independent", im).set("independentSe", is));
      }
      return Value::object()
          .set("score", scores[0])
          .set("bump", q.bump)
          .set("replications", q.replications)
          .set("learners", static_cast<double>(plans[0].agents))
          .set("varianceReduction", count ? ratio / count : 0.0)
          .set("levers", levers)
          .set("ranked", Value::Numbers(rankedBy(crnMeans)));
    };
    return m;
  }();
  return &model;
}

const kernel::Model* gagneModel() {
  static const kernel::Model model = [] {
    kernel::Model m = base("gagne", "Gagné's nine events of instruction: the cost of leaving each one out");
    m.compile = [](const Value& spec) {
      std::array<double, 9> all;
      all.fill(1.0);
      std::vector<Job> jobs = {{gagneDesign(all), seedOf(spec)}};
      for (int k = 0; k < 9; ++k) {
        auto missing = all;
        missing[k] = 0.0;
        jobs.push_back({gagneDesign(missing), seedOf(spec)});
      }
      std::array<double, 9> lecture{};
      lecture[3] = 1.0;
      jobs.push_back({gagneDesign(lecture), seedOf(spec)});
      return std::vector<Plan>{compileClassroom("gagne", populationOf(spec.sub("population")), courseOf(spec.sub("course")), learnersOf(spec, 500), jobs, false, false)};
    };
    m.summarise = [](const Value& spec, const std::vector<Plan>& plans, const std::vector<Output>& out) {
      const auto scores = delayedScores(plans[0], out[0], courseOf(spec.sub("course")));
      Value events = Value::array();
      for (int k = 0; k < 9; ++k) events.push(Value::object().set("event", kGagneEvents[k]).set("withoutIt", scores[1 + k]));
      std::array<double, 9> all;
      all.fill(1.0);
      return Value::object().set("full", scores[0]).set("presentOnly", scores[10]).set("events", events).set("design", designValue(gagneDesign(all)));
    };
    return m;
  }();
  return &model;
}

}  // namespace syn::models
