// Memory, forgetting and practice (cognitivism) in the space.
//
// A memory trace is a point of the cognitive channel at angle φ from the learner; recall is its
// chance cos²(φ/2). Forgetting is the rotor in the (N, A) plane that turns the trace away from the
// learner as time passes: the angle after Δt days at stability S is the one at which the chance is
// the power law R = (1 + 19/81 · Δt/S)^{−1/2}. A study event at a retrievability R multiplies S
// (desirable difficulty: the harder the recall, the larger the gain); a retrieval attempt
// succeeds when the trace lies in the cap of a uniform draw; a failed attempt with feedback
// relearns from a weaker base. Jobs: six study plans (and one job of the gain curves); agents:
// learners whose encoding depth is log-logistic around the plan's.
#include <string>

#include "syn/models/models.hpp"
#include "syn/kernel/agent.hpp"

SYN_AGENT(memory_agent,
  let horizon = u32(P(0u));
  let testDay = P(1u);
  let restudyFloor = P(3u);
  let restudyGain = P(4u);
  let retrievalFloor = P(5u);
  let retrievalGain = P(6u);
  let failure = P(7u);
  let minStability = P(8u);
  if (J(18u) > f32(0.5)) {
    let r = J(19u);
    stat(0u, f32(1) + restudyFloor + restudyGain * (f32(1) - r));
    stat(1u, f32(1) + retrievalFloor + retrievalGain * (f32(1) - r));
    return;
  }
  let depth = J(17u) * exp(P(9u) * kLogisticUnit * logisticOf(U(0u)));
  var stability = P(2u) * depth;
  var last = f32(0);
  let count = u32(J(0u));
  var next = 0u;
  for (var d = 0u; d <= horizon; d = d + 1u) {
    let day = f32(d);
    while (next < count && J(1u + next) <= day) {
      let t = J(1u + next);
      let mode = J(9u + next);
      let r = select(f32(1), f32(1) / sqrt(f32(1) + f32(19) / f32(81) * (t - last) / stability), t > last);
      let trace = rotate(pole(), kN, kA, f32(2) * acos(sqrt(r)));
      let restudy = mode < f32(0.5);
      let gainRestudy = f32(1) + restudyFloor + restudyGain * (f32(1) - r);
      let gainRetrieval = f32(1) + retrievalFloor + retrievalGain * (f32(1) - r);
      let recalled = inside(trace, chanceCap(U(1u + next)));
      if (!restudy) {
        stat(66u, f32(1));
        stat(67u, f32(recalled));
      }
      if (restudy || recalled) {
        stability = stability * select(gainRetrieval, gainRestudy, restudy);
        last = t;
      } else if (mode > f32(1.5)) {
        stability = max(minStability, failure * stability) * gainRestudy;
        last = t;
      }
      next = next + 1u;
    }
    let r = select(f32(1), f32(1) / sqrt(f32(1) + f32(19) / f32(81) * (day - last) / stability), day > last);
    let trace = rotate(pole(), kN, kA, f32(2) * acos(sqrt(r)));
    stat(d, chance(trace));
    quant(d, chance(trace));
    if (day == testDay) {
      stat(64u, chance(trace));
      stat(65u, stability);
    }
  }
)

namespace syn::models {

namespace {
struct Event {
  double day;
  int mode;  // 0 restudy, 1 retrieval, 2 retrieval with feedback
};
struct StudyPlan {
  std::string name;
  std::vector<Event> events;
};
const char* modeName(int m) { return m == 0 ? "restudy" : (m == 1 ? "retrieval" : "retrieval + feedback"); }
std::string num(double x) {
  std::string s = std::to_string(x);
  s.erase(s.find_last_not_of('0') + 1);
  if (!s.empty() && s.back() == '.') s.pop_back();
  return s;
}
StudyPlan massed(int mode, int count, double day) {
  StudyPlan p{"massed " + std::string(modeName(mode)) + " ×" + std::to_string(count) + " on day " + num(day), {}};
  for (int i = 0; i < count; ++i) p.events.push_back({day + 0.002 * (i + 1), mode});
  return p;
}
StudyPlan spaced(int mode, int count, double gap) {
  StudyPlan p{"spaced " + std::string(modeName(mode)) + " ×" + std::to_string(count) + " every " + num(gap) + "d", {}};
  for (int i = 0; i < count; ++i) p.events.push_back({gap * (i + 1), mode});
  return p;
}
StudyPlan expanding(int mode, int count, double first, double ratio) {
  StudyPlan p{"expanding " + std::string(modeName(mode)) + " ×" + std::to_string(count) + " from " + num(first) + "d", {}};
  double t = 0.0, gap = first;
  for (int i = 0; i < count; ++i) {
    t += gap;
    gap *= ratio;
    p.events.push_back({t, mode});
  }
  return p;
}
std::vector<StudyPlan> plansOf(const Value& c) {
  const int reviews = std::min(8, c.integer("reviews", 4));
  const double gap = c.num("gap", 3), testDay = c.num("testDay", 30);
  std::vector<StudyPlan> plans = {massed(0, reviews, 0), massed(1, reviews, 0), spaced(0, reviews, gap), spaced(2, reviews, gap),
                                  expanding(2, reviews, 1, 2), massed(0, reviews, c.num("cramDay", testDay - 1))};
  plans.back().name = "cram: massed restudy the day before the test";
  return plans;
}
}  // namespace

const kernel::Model* memoryModel() {
  static const kernel::Model model = [] {
    kernel::Model m;
    m.name = "memory";
    m.title = "Forgetting, retrieval practice and spacing: memory traces turned by rotors";
    m.maxStats = 68;
    SYN_USE_AGENT(m, memory_agent);
    m.compile = [](const Value& spec) {
      const Value& c = spec.sub("memory");
      const int horizon = std::min(63, static_cast<int>(std::ceil(c.num("horizon", 45))));
      const std::vector<double> params = {double(horizon), std::floor(c.num("testDay", 30)), c.num("initialStability", 1.0), 0.05, 0.6, 0.30, 5.0,
                                          0.7, 0.5, c.num("heterogeneity", 0.3)};
      std::vector<std::vector<double>> rows;
      for (const auto& p : plansOf(c)) {
        std::vector<double> r(20, 0.0);
        r[0] = static_cast<double>(p.events.size());
        for (std::size_t k = 0; k < p.events.size(); ++k) {
          r[1 + k] = p.events[k].day;
          r[9 + k] = p.events[k].mode;
        }
        r[17] = 1.0;  // encoding depth
        rows.push_back(r);
      }
      Plan plans = makePlan("memory", static_cast<std::uint32_t>(c.integer("learners", 400)), 68, params, rows, seedOf(spec),
                            static_cast<std::uint32_t>(horizon + 1));
      std::vector<std::vector<double>> gains;
      for (double r : grid(0.05, 1.0, 40)) {
        std::vector<double> row(20, 0.0);
        row[18] = 1;
        row[19] = r;
        gains.push_back(row);
      }
      return std::vector<Plan>{plans, makePlan("memory", 1, 2, params, gains, seedOf(spec))};
    };
    m.summarise = [](const Value& spec, const std::vector<Plan>& plans, const std::vector<Output>& out) {
      const Value& c = spec.sub("memory");
      const Plan& p = plans[0];
      const Output& o = out[0];
      const int horizon = static_cast<int>(p.params[0]);
      const auto study = plansOf(c);
      std::vector<double> days;
      for (int d = 0; d <= horizon; ++d) days.push_back(d);
      Value list = Value::array();
      for (std::uint32_t j = 0; j < p.jobs; ++j) {
        Value x = Value::object();
        std::vector<double> q10, q90, events;
        for (int d = 0; d <= horizon; ++d) {
          q10.push_back(o.quantiles[(j * p.quantiles + d) * 3]);
          q90.push_back(o.quantiles[(j * p.quantiles + d) * 3 + 2]);
        }
        for (const auto& e : study[j].events) events.push_back(e.day);
        const double attempts = o.mean(p, j, 66);
        x.set("name", study[j].name).set("days", Value::Numbers(days)).set("recall", Value::Numbers(o.range(p, j, 0, horizon + 1)));
        x.set("q10", Value::Numbers(q10)).set("q90", Value::Numbers(q90)).set("recallAtTest", o.mean(p, j, 64));
        x.set("stability", o.mean(p, j, 65)).set("successRate", attempts > 0 ? o.mean(p, j, 67) / attempts : 1.0);
        x.set("events", Value::Numbers(events));
        list.push(x);
      }
      Value r = Value::object();
      r.set("plans", list).set("testDay", p.params[1]);
      r.set("gainR", Value::Numbers(grid(0.05, 1.0, 40)));
      r.set("gainRestudy", Value::Numbers(column(plans[1], out[1], 0))).set("gainRetrieval", Value::Numbers(column(plans[1], out[1], 1)));
      return r;
    };
    return m;
  }();
  return &model;
}

}  // namespace syn::models
