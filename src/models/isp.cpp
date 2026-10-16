// Kuhlthau's Information Search Process in the space, with zones of intervention.
//
// A student moves through seven stages (initiation, selection, exploration, formulation,
// collection, presentation, assessment). Trait library anxiety is a random share of the affective
// channel; the uncertainty felt in a stage is that stage's typical feeling (a point) scaled by the
// anxiety, eased by the instructor's support and by a one-shot session in its zone: each scaling
// is a mixture of the point (nudge). The day's work is twice the chance of a random level around
// the student's effort, slowed by the felt uncertainty; progress through the stage is a point
// raised towards completion, and the stage is done when it would pass it.
// Jobs: one per (support, relevance, session day, targeted stage, strength); agents: students.
#include "syn/models/models.hpp"
#include "syn/kernel/agent.hpp"

SYN_AGENT(isp_agent,
  let deadline = u32(P(0u));
  let horizon = u32(P(1u));
  let amplification = P(4u);
  let interference = P(5u);
  let support = clamp(J(0u), f32(0), f32(1));
  let relevance = clamp(J(1u), f32(0), f32(1));
  let sessionDay = J(2u);
  let aim = u32(J(3u));
  let strength = clamp(J(4u), f32(0), f32(1));
  var anxiety = sampleShare(U(0u), P(2u), P(3u), kB);
  let effort = sampleLevel(U(1u), f32(0), P(6u), kB);
  let boost = f32(1) + f32(0.3) * relevance;
  var stage = 0u;
  var progress = atChance(f32(0), kA);
  var help = f32(0);
  var formulationDay = f32(-1);
  var doneDay = f32(-1);
  for (var d = 0u; d < horizon; d = d + 1u) {
    if (f32(d) == sessionDay && stage <= aim) {
      help = strength * pow(f32(0.7), f32(aim - stage));
      stat(651u, f32(1));
    }
    if (stage < 7u) {
      let zone = select(f32(0), help, stage == aim);
      let typical = P(8u + stage);
      let work = P(15u + stage);
      var felt = atChance(typical, kB);
      felt = nudge(felt, chance(felt) * amplification * (f32(2) * chance(anxiety) - f32(1)));
      felt = lower(felt, f32(0.5) * support);
      felt = lower(felt, f32(0.4) * zone);
      stat(d, chance(felt));
      stat(64u + d, f32(1));
      stat(128u + stage * 64u + d, f32(1));
      stat(640u + stage, f32(1));
      let today = f32(2) * chance(sampleLevel(U(2u + d), logit(effort), P(7u), kB)) * boost;
      let step = today / (f32(1) + interference * chance(felt)) / (work * (f32(1) - f32(0.5) * zone));
      anxiety = lower(anxiety, f32(0.03) * support);
      if (step >= f32(1) - chance(progress)) {
        progress = atChance(f32(0), kA);
        stage = stage + 1u;
        if (stage == 3u) { formulationDay = f32(d + 1u); }
        if (stage == 6u) { doneDay = f32(d + 1u); }
      } else {
        progress = nudge(progress, step);
      }
    } else {
      stat(128u + 7u * 64u + d, f32(1));
    }
  }
  if (doneDay >= f32(0) && doneDay <= f32(deadline)) {
    stat(647u, f32(1));
    if (formulationDay >= f32(0) && formulationDay <= f32(0.6) * f32(deadline)) { stat(648u, f32(1)); }
  }
  if (doneDay >= f32(0)) {
    stat(649u, doneDay);
    stat(650u, f32(1));
  }
)

namespace syn::models {

namespace {
struct Stage {
  const char* name;
  const char* feeling;
  double uncertainty, work;
};
const Stage kStages[7] = {
    {"initiation", "uncertainty, apprehension", 0.60, 2.0},
    {"selection", "optimism", 0.40, 2.0},
    {"exploration", "confusion, frustration, doubt", 0.85, 8.0},
    {"formulation", "clarity", 0.35, 3.0},
    {"collection", "sense of direction, confidence", 0.25, 6.0},
    {"presentation", "relief; satisfaction or disappointment", 0.15, 3.0},
    {"assessment", "sense of accomplishment (reflection)", 0.10, 1.0},
};
}  // namespace

const kernel::Model* ispModel() {
  static const kernel::Model model = [] {
    kernel::Model m;
    m.name = "isp";
    m.title = "Kuhlthau's Information Search Process: felt uncertainty, support and zones of intervention";
    m.maxStats = 652;
    SYN_USE_AGENT(m, isp_agent);
    m.compile = [](const Value& spec) {
      const Value& c = spec.sub("isp");
      const int deadline = std::min(42, c.integer("deadline", 42));
      const int horizon = deadline + deadline / 2;
      std::vector<double> params = {double(deadline), double(horizon), c.num("anxiety", 0.5), c.num("anxietySd", 0.2),
                                    c.num("amplification", 0.8), c.num("interference", 1.6), c.num("effortSd", 0.5), c.num("dailySd", 0.5)};
      for (const auto& s : kStages) params.push_back(s.uncertainty);
      for (const auto& s : kStages) params.push_back(s.work);
      std::vector<std::vector<double>> rows;
      const double relevance = c.num("relevance", 0.0), strength = c.num("strength", 0.6);
      for (double support : c.nums("supports", {0.0, 0.5, 1.0})) rows.push_back({support, relevance, -1, 0, 0});
      // Zones of intervention: a one-shot session on each day targeting each stage.
      if (c.flag("zones", false))
        for (int stage = 1; stage <= 4; ++stage)
          for (double day : c.nums("sessionDays", {2, 5, 8, 12, 16, 20, 25})) rows.push_back({0.0, relevance, day, double(stage), strength});
      return std::vector<Plan>{makePlan("isp", static_cast<std::uint32_t>(c.integer("students", 400)), 652, params, rows, seedOf(spec))};
    };
    m.summarise = [](const Value& spec, const std::vector<Plan>& plans, const std::vector<Output>& out) {
      const Plan& p = plans[0];
      const Output& o = out[0];
      const Value& c = spec.sub("isp");
      const int horizon = static_cast<int>(p.params[1]);
      const auto supports = c.nums("supports", {0.0, 0.5, 1.0});
      auto job = [&](std::uint32_t j) {
        Value x = Value::object();
        std::vector<double> days, uncertainty;
        for (int d = 0; d < horizon; ++d) {
          days.push_back(d);
          const double w = o.mean(p, j, 64 + d);
          uncertainty.push_back(w > 0 ? o.mean(p, j, d) / w : 0.0);
        }
        Value shares = Value::array();
        for (int s = 0; s < 8; ++s) shares.push(Value::Numbers(o.range(p, j, 128 + s * 64, horizon)));
        x.set("days", Value::Numbers(days)).set("uncertainty", Value::Numbers(uncertainty)).set("stageShare", shares);
        x.set("daysInStage", Value::Numbers(o.range(p, j, 640, 7)));
        x.set("completedOnTime", o.mean(p, j, 647)).set("satisfied", o.mean(p, j, 648)).set("helped", o.mean(p, j, 651));
        const double done = o.mean(p, j, 650);
        x.set("meanCompletionDay", done > 0 ? o.mean(p, j, 649) / done : 0.0);
        return x;
      };
      Value r = Value::object();
      Value runs = Value::array();
      for (std::uint32_t j = 0; j < supports.size(); ++j) runs.push(job(j).set("support", supports[j]));
      r.set("isp", runs);
      Value stages = Value::array();
      for (const auto& s : kStages) stages.push(Value::object().set("name", s.name).set("feeling", s.feeling).set("uncertainty", s.uncertainty));
      r.set("stages", stages).set("deadline", static_cast<int>(p.params[0]));
      if (c.flag("zones", false)) {
        const auto days = c.nums("sessionDays", {2, 5, 8, 12, 16, 20, 25});
        Value zones = Value::array();
        std::uint32_t j = static_cast<std::uint32_t>(supports.size());
        for (int stage = 1; stage <= 4; ++stage) {
          std::vector<double> onTime, helped;
          for (std::size_t k = 0; k < days.size(); ++k, ++j) {
            onTime.push_back(o.mean(p, j, 647));
            helped.push_back(o.mean(p, j, 651));
          }
          zones.push(Value::object().set("stage", kStages[stage].name).set("onTime", Value::Numbers(onTime)).set("helped", Value::Numbers(helped)));
        }
        r.set("sessionDays", Value::Numbers(days)).set("zones", zones).set("noSession", o.mean(p, 0, 647));
      }
      return r;
    };
    return m;
  }();
  return &model;
}

}  // namespace syn::models
