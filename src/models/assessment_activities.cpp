// Assessing learning (chapter 9): choosing assessment activities, and what assessment, grades,
// self-report and satisfaction say about learning, in the space.
//
// Each activity's evidence speaks to one level of Bloom's taxonomy; aligned with an outcome at
// level o, evidence at level m keeps 0.4^(o − m) of its meaning below the outcome (it overstates
// achievement) and 0.75^(m − o) above it; self-report keeps 30%. The information is
// reach × (1 − noise), the fit to the audience is lost as the writing and test-likeness exceed
// what the audience tolerates, and a sixth of the session is a fair price in time.
// Signals: a learner's learning is a level; a direct assessment reads it with noise; a grade mixes
// it with attendance, effort and conduct; self-report regresses to a flattering mean (the least
// skilled overestimate most), a mixture of the learner's point towards it; satisfaction mostly
// reflects enjoyment. The kernel accumulates the sums behind each correlation.
#include <string>

#include "syn/models/models.hpp"

// The agent program, in WGSL.
constexpr const char* activities_agent = R"wgsl(
  if (J(0u) < f32(0.5)) {
    let outcome = J(1u);
    let writing = J(2u);
    let tests = J(3u);
    let session = J(4u);
    let a = u32(J(5u)) * 10u;
    let minutes = P(a);
    let direct = P(a + 1u) > f32(0.5);
    let measures = P(a + 4u);
    let noise = P(a + 5u);
    let reach = P(a + 6u);
    let writingLoad = P(a + 7u);
    let testLike = P(a + 8u);
    var alignment = f32(1);
    if (measures < outcome) { alignment = pow(f32(0.4), outcome - measures); }
    if (measures > outcome) { alignment = pow(f32(0.75), measures - outcome); }
    if (!direct) { alignment = alignment * f32(0.3); }
    let information = reach * (f32(1) - noise);
    let fit = clamp(f32(1) - f32(1.2) * max(f32(0), writingLoad - writing) - f32(1.2) * max(f32(0), testLike - tests), f32(0), f32(1));
    let cost = minutes / session;
    stat(0u, alignment);
    stat(1u, information);
    stat(2u, fit);
    stat(3u, cost);
    stat(4u, alignment * information * fit - max(f32(0), cost - f32(0.15)));
  } else {
    let learning = sampleLevel(U(0u), f32(0), f32(1), kA);
    let l = logit(learning);
    let attendance = kLogisticUnit * logisticOf(U(1u));
    let effort = f32(0.3) * l + f32(0.95) * kLogisticUnit * logisticOf(U(2u));
    let conduct = kLogisticUnit * logisticOf(U(3u));
    let skill = atLogit(l, kA);
    let direct = l + f32(0.45) * kLogisticUnit * logisticOf(U(4u));
    let grade = f32(0.45) * l + f32(0.3) * attendance + f32(0.35) * effort + f32(0.2) * conduct + f32(0.3) * kLogisticUnit * logisticOf(U(5u));
    let report = chance(nudge(blend(skill, atChance(f32(0.6), kA), f32(0.6)), f32(0.05) + f32(0.12) * kLogisticUnit * logisticOf(U(6u))));
    let satisfaction = f32(0.25) * l + kLogisticUnit * logisticOf(U(7u));
    stat(0u, l);
    stat(1u, l * l);
    var signals : array<f32, 4>;
    signals[0] = direct;
    signals[1] = grade;
    signals[2] = report;
    signals[3] = satisfaction;
    for (var k = 0u; k < 4u; k = k + 1u) {
      stat(2u + 3u * k, signals[k]);
      stat(3u + 3u * k, signals[k] * signals[k]);
      stat(4u + 3u * k, signals[k] * l);
    }
  }
)wgsl";

namespace syn::models {

namespace {
struct Activity {
  const char* name;
  double minutes, direct, formal, authentic, measures, noise, reach, writing, testLike, after;
};
// minutes, direct, formal, authentic, Bloom level, noise, reach, writing load, test-likeness, after class
const Activity kActivities[13] = {
    {"quick poll (\"got it?\")", 1, 0, 0, 0, 0, 0.30, 0.8, 0.0, 0.1, 0},
    {"observing practice", 0, 1, 0, 1, 2, 0.30, 0.4, 0.0, 0.0, 0},
    {"worksheet", 10, 1, 1, 1, 2, 0.12, 1.0, 0.4, 0.3, 0},
    {"task / demonstration", 8, 1, 1, 1, 2, 0.15, 0.6, 0.0, 0.2, 0},
    {"pre/post test (fixed choice)", 8, 1, 1, 0, 0, 0.12, 1.0, 0.1, 1.0, 0},
    {"graphic organizer", 6, 1, 1, 0, 1, 0.15, 1.0, 0.2, 0.3, 0},
    {"pro/con chart", 8, 1, 1, 1, 3, 0.18, 1.0, 0.4, 0.2, 0},
    {"annotated bibliography", 0, 1, 1, 1, 4, 0.10, 0.6, 0.9, 0.4, 1},
    {"exit ticket", 3, 1, 1, 0, 1, 0.22, 1.0, 0.4, 0.2, 0},
    {"directed paraphrasing", 4, 1, 1, 1, 1, 0.18, 1.0, 0.6, 0.2, 0},
    {"one-sentence summary", 3, 1, 1, 0, 1, 0.22, 1.0, 0.5, 0.2, 0},
    {"application card", 4, 1, 1, 1, 2, 0.22, 1.0, 0.5, 0.2, 0},
    {"self-assessment questionnaire", 3, 0, 1, 0, 0, 0.10, 1.0, 0.2, 0.3, 0},
};
struct Audience {
  const char* name;
  double writing, tests;
};
const Audience kAudiences[4] = {{"undergraduates", 0.7, 0.6}, {"high school students", 0.5, 0.8}, {"older adults (public library)", 0.4, 0.1},
                                {"English-language learners", 0.2, 0.5}};
const char* kLevels[6] = {"remember", "understand", "apply", "analyze", "evaluate", "create"};
}  // namespace

const kernel::Model* assessmentActivitiesModel() {
  static const kernel::Model model = [] {
    kernel::Model m;
    m.name = "assessmentActivities";
    m.title = "Choosing assessment activities by alignment, audience and time; assessment vs grades, self-report and satisfaction";
    m.maxStats = 14;
    m.agentSource = activities_agent;
    m.compile = [](const Value& spec) {
      const Value& c = spec.sub("assessment");
      const double minutes = c.num("minutes", 45.0);
      if (minutes <= 0) throw std::invalid_argument("assessment: the session needs a positive length");
      std::vector<double> params;
      for (const auto& a : kActivities)
        params.insert(params.end(), {a.minutes, a.direct, a.formal, a.authentic, a.measures, a.noise, a.reach, a.writing, a.testLike, a.after});
      std::vector<std::vector<double>> rows;
      for (const auto& au : kAudiences)
        for (int level = 0; level < 6; ++level)
          for (int a = 0; a < 13; ++a) rows.push_back({0, double(level), au.writing, au.tests, minutes, double(a)});
      return std::vector<Plan>{makePlan("assessmentActivities", 1, 5, params, rows, seedOf(spec)),
                               makePlan("assessmentActivities", static_cast<std::uint32_t>(c.integer("learners", 2000)), 14, params, {{1, 0, 0, 0, 0, 0}},
                                        seedOf(spec))};
    };
    m.summarise = [](const Value&, const std::vector<Plan>& plans, const std::vector<Output>& out) {
      Value r = Value::object();
      Value acts = Value::array();
      for (const auto& a : kActivities)
        acts.push(Value::object()
                      .set("name", a.name)
                      .set("minutes", a.minutes)
                      .set("direct", a.direct > 0.5)
                      .set("formal", a.formal > 0.5)
                      .set("authentic", a.authentic > 0.5)
                      .set("measures", kLevels[static_cast<int>(a.measures)])
                      .set("noise", a.noise)
                      .set("reach", a.reach)
                      .set("afterSession", a.after > 0.5));
      r.set("activities", acts).set("levels", std::vector<std::string>(kLevels, kLevels + 6));
      const Plan& p = plans[0];
      Value audiences = Value::array();
      for (std::uint32_t au = 0; au < 4; ++au) {
        Value byLevel = Value::array(), parts = Value::array();
        for (std::uint32_t level = 0; level < 6; ++level) {
          std::vector<double> v, al, inf, fit, cost;
          for (std::uint32_t a = 0; a < 13; ++a) {
            const std::uint32_t j = (au * 6 + level) * 13 + a;
            v.push_back(out[0].mean(p, j, 4));
            al.push_back(out[0].mean(p, j, 0));
            inf.push_back(out[0].mean(p, j, 1));
            fit.push_back(out[0].mean(p, j, 2));
            cost.push_back(out[0].mean(p, j, 3));
          }
          byLevel.push(Value::Numbers(v));
          parts.push(Value::object().set("alignment", Value::Numbers(al)).set("information", Value::Numbers(inf)).set("fit", Value::Numbers(fit))
                         .set("timeCost", Value::Numbers(cost)));
        }
        Value top = Value::array();
        for (const auto& v : byLevel.asArray()) {
          auto ranked = rankedBy(v.asNumbers());
          ranked.resize(std::min<std::size_t>(3, ranked.size()));
          top.push(Value::Numbers(ranked));
        }
        audiences.push(Value::object().set("name", kAudiences[au].name).set("value", byLevel).set("parts", parts).set("top", top));
      }
      r.set("audiences", audiences);
      const Plan& q = plans[1];
      const double n = q.agents;
      auto S = [&](std::uint32_t i) { return out[1].mean(q, 0, i) * n; };
      const double meanL = S(0) / n, varL = S(1) / n - meanL * meanL;
      auto corr = [&](std::uint32_t k) {
        const double sx = S(2 + 3 * k) / n, sxx = S(3 + 3 * k) / n, sxl = S(4 + 3 * k) / n;
        const double cov = sxl - sx * meanL, varX = sxx - sx * sx;
        return varX > 0 && varL > 0 ? cov / std::sqrt(varX * varL) : 0.0;
      };
      r.set("signals", Value::object().set("direct", corr(0)).set("grade", corr(1)).set("selfReport", corr(2)).set("satisfaction", corr(3)));
      return r;
    };
    return m;
  }();
  return &model;
}

}  // namespace syn::models
