// A classroom point system (token economy) in the space.
//
// A student engages on a day when the point of engagement lies in the cap of a uniform draw.
// That point is the baseline level of the affective channel dilated by the student's intrinsic
// interest (a random level), their habit and, while points are given, the incentive (plus a
// gamification bonus per level reached): dilations about the learner compose by adding logits.
// Rewarded engagement raises the habit point and erodes intrinsic interest (a dilation back
// towards zero: the undermining effect); unrewarded engagement after the points stop lowers the
// habit. Agents: students; one job.
#include "syn/models/models.hpp"
#include "syn/kernel/agent.hpp"

SYN_AGENT(token_agent,
  let days = u32(P(0u));
  let start = u32(P(1u));
  let stop = u32(P(2u));
  let interest0 = sampleLevel(U(0u), P(4u), P(5u), kB);
  let high = logit(interest0) > P(4u);
  var interest = interest0;
  var habit = atChance(f32(0), kB);
  var points = f32(0);
  for (var d = 0u; d < days; d = d + 1u) {
    let rewards = d >= start && d < stop;
    let level = floor(points / P(8u));
    let incentive = select(f32(0), P(6u) + P(7u) * level, rewards);
    let engage = shift(shift(atLogit(P(3u), kB), logit(interest)), P(10u) * chance(habit) + incentive);
    let prob = chance(engage);
    stat(d, prob);
    stat(64u + d, prob * f32(high));
    if (inside(engage, chanceCap(U(1u + d)))) {
      if (rewards) {
        points = points + f32(1);
        habit = raise(habit, P(9u));
        let x = logit(interest);
        if (x > f32(0)) { interest = shift(interest, -min(x, P(12u) * (f32(1) + x))); }
      } else {
        habit = lower(habit, P(11u));
      }
    }
    stat(128u + d, points);
    stat(192u + d, logit(interest));
  }
  stat(256u, f32(high));
)

namespace syn::models {

const kernel::Model* tokenEconomyModel() {
  static const kernel::Model model = [] {
    kernel::Model m;
    m.name = "tokenEconomy";
    m.title = "A classroom point system: engagement, habit and the undermining of intrinsic interest";
    m.maxStats = 257;
    SYN_USE_AGENT(m, token_agent);
    m.compile = [](const Value& spec) {
      const Value& c = spec.sub("tokens");
      const int days = std::min(64, c.integer("days", 60));
      const std::vector<double> params = {double(days), double(c.integer("rewardStart", 15)), double(c.integer("rewardEnd", 40)),
                                          c.num("baseline", -0.6), c.num("intrinsicMean", 0.0), c.num("intrinsicSd", 1.0),
                                          c.num("incentive", 1.2), c.num("levelBonus", 0.0), double(c.integer("levelPoints", 10)),
                                          c.num("habitRate", 0.04), c.num("habitMax", 1.0), c.num("habitDecay", 0.03), c.num("undermining", 0.004)};
      return std::vector<Plan>{makePlan("tokenEconomy", static_cast<std::uint32_t>(c.integer("students", 200)), 257, params, {{0.0}}, seedOf(spec))};
    };
    m.summarise = [](const Value&, const std::vector<Plan>& plans, const std::vector<Output>& out) {
      const Plan& p = plans[0];
      const Output& o = out[0];
      const int days = static_cast<int>(p.params[0]), start = static_cast<int>(p.params[1]), stop = static_cast<int>(p.params[2]);
      const double highShare = o.mean(p, 0, 256);
      std::vector<double> eng = o.range(p, 0, 0, days), high, low;
      for (int d = 0; d < days; ++d) {
        const double h = o.mean(p, 0, 64 + d);
        high.push_back(highShare > 0 ? h / highShare : 0.0);
        low.push_back(highShare < 1 ? (eng[d] - h) / (1.0 - highShare) : 0.0);
      }
      double sums[3] = {0, 0, 0};
      int counts[3] = {0, 0, 0};
      for (int d = 0; d < days; ++d) {
        const int phase = d < start ? 0 : (d < stop ? 1 : 2);
        sums[phase] += eng[d];
        ++counts[phase];
      }
      Value r = Value::object();
      r.set("engagement", Value::Numbers(eng)).set("low", Value::Numbers(low)).set("high", Value::Numbers(high));
      r.set("points", Value::Numbers(o.range(p, 0, 128, days))).set("intrinsic", Value::Numbers(o.range(p, 0, 192, days)));
      r.set("before", counts[0] ? sums[0] / counts[0] : 0.0).set("during", counts[1] ? sums[1] / counts[1] : 0.0);
      r.set("after", counts[2] ? sums[2] / counts[2] : 0.0).set("rewardStart", start).set("rewardEnd", stop);
      return r;
    };
    return m;
  }();
  return &model;
}

}  // namespace syn::models
