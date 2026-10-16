// Operant conditioning (Skinner) in the space: schedules of reinforcement and extinction.
//
// The probability of the response is a point of the cognitive channel. Each opportunity the
// learner responds when the point lies in the cap of a uniform draw u (chance ≥ u). A reinforced
// response raises the point towards the learner, raise(p, a); an unreinforced one moves it back
// towards the operant level by a mixture, blend(p, baseline, e·surprise), where the surprise of a
// run of r unreinforced responses grows with the expected rate of reinforcement r̂ (Laplace):
// (1 − (1 − r̂)^r)^k. Partial reinforcement keeps r̂ low, so extinction is slow (the partial
// reinforcement extinction effect). Jobs: CRF, FR n, VR n, FI n, VI n; agents: learners.
#include "syn/models/models.hpp"
#include "syn/kernel/agent.hpp"

SYN_AGENT(operant_agent,
  let kind = J(0u);
  let n = max(f32(1), J(1u));
  let training = u32(P(0u));
  let steps = training + u32(P(1u));
  let every = u32(P(2u));
  let rate = P(3u);
  let decay = P(4u);
  let sharp = P(5u);
  let baseline = atChance(P(6u), kA);
  let whole = u32(floor(n + f32(0.5)));
  let spread = f32(2) * n - f32(1);
  var p = baseline;
  var responses = f32(0);
  var reinforcers = f32(0);
  var unrewarded = f32(0);
  var count = 0u;
  let r0 = U(100000u);
  var requirement = select(whole, 1u + u32(floor(r0 * spread)), kind > f32(1.5));
  var availableAt = select(whole, 1u + u32(floor(r0 * spread)), kind > f32(3.5));
  var cumR = f32(0);
  var cumS = f32(0);
  var peak = f32(0);
  var acquired = f32(training);
  var extinct = f32(steps - training);
  var acqDone = false;
  var extDone = false;
  for (var t = 0u; t < steps; t = t + 1u) {
    let inTraining = t < training;
    if (inside(p, chanceCap(U(2u * t)))) {
      cumR = cumR + f32(1);
      var reinforced = false;
      if (inTraining) {
        if (kind < f32(0.5)) {
          reinforced = true;
        } else if (kind < f32(2.5)) {
          count = count + 1u;
          reinforced = count >= requirement;
        } else {
          reinforced = t >= availableAt;
        }
      }
      if (reinforced) {
        p = raise(p, rate);
        responses = responses + f32(1);
        reinforcers = reinforcers + f32(1);
        unrewarded = f32(0);
        cumS = cumS + f32(1);
        count = 0u;
        let r = U(2u * t + 1u);
        requirement = select(whole, 1u + u32(floor(r * spread)), kind > f32(1.5));
        availableAt = t + select(whole, 1u + u32(floor(r * spread)), kind > f32(3.5));
      } else {
        let expected = (reinforcers + f32(1)) / (responses + f32(2));
        unrewarded = unrewarded + f32(1);
        let surprise = pow(f32(1) - pow(f32(1) - expected, unrewarded), sharp);
        p = blend(p, baseline, decay * surprise);
        responses = responses + f32(1);
      }
    }
    let pc = chance(p);
    if (inTraining) {
      peak = max(peak, pc);
      if (!acqDone && pc >= f32(0.8)) { acquired = f32(t + 1u); acqDone = true; }
    } else if (!extDone && pc < f32(0.5) * peak) {
      extinct = f32(t - training + 1u);
      extDone = true;
    }
    if ((t + 1u) % every == 0u) {
      let i = t / every;
      stat(i, pc);
      stat(128u + i, cumR);
      stat(256u + i, cumS);
    }
  }
  stat(384u, acquired);
  stat(385u, extinct);
  stat(386u, cumR);
  stat(387u, cumS);
)

namespace syn::models {

const kernel::Model* operantModel() {
  static const kernel::Model model = [] {
    kernel::Model m;
    m.name = "operant";
    m.title = "Operant conditioning: schedules of reinforcement and the partial reinforcement extinction effect";
    m.maxStats = 388;
    SYN_USE_AGENT(m, operant_agent);
    m.compile = [](const Value& spec) {
      const Value& c = spec.sub("operant");
      const int training = c.integer("training", 300), extinction = c.integer("extinction", 300);
      int every = 1;
      while ((training + extinction) / every > 128) ++every;
      const double n = c.num("ratio", 5);
      const std::vector<double> params = {double(training), double(extinction), double(every), c.num("acquisitionRate", 0.12),
                                          c.num("extinctionRate", 0.15), c.num("surpriseExponent", 6.0), c.num("baseline", 0.05)};
      const std::vector<std::vector<double>> rows = {{0, 1}, {1, n}, {2, n}, {3, n}, {4, n}};
      return std::vector<Plan>{makePlan("operant", static_cast<std::uint32_t>(c.integer("learners", 200)), 388, params, rows, seedOf(spec))};
    };
    m.summarise = [](const Value& spec, const std::vector<Plan>& plans, const std::vector<Output>& out) {
      const Plan& p = plans[0];
      const Output& o = out[0];
      const int training = static_cast<int>(p.params[0]), every = static_cast<int>(p.params[2]);
      const int points = (training + static_cast<int>(p.params[1])) / every;
      const double n = spec.sub("operant").num("ratio", 5);
      const auto label = [&](int k) {
        const std::string v = std::to_string(static_cast<int>(std::lround(n)));
        const char* names[] = {"continuous (CRF)", "fixed ratio FR", "variable ratio VR", "fixed interval FI", "variable interval VI"};
        return k == 0 ? std::string(names[0]) : std::string(names[k]) + v;
      };
      Value runs = Value::array();
      std::vector<double> steps;
      for (int i = 0; i < points; ++i) steps.push_back((i + 1) * every);
      for (std::uint32_t j = 0; j < p.jobs; ++j) {
        Value x = Value::object();
        x.set("schedule", label(static_cast<int>(j)));
        x.set("probability", Value::Numbers(o.range(p, j, 0, points)));
        x.set("cumulativeResponses", Value::Numbers(o.range(p, j, 128, points)));
        x.set("cumulativeReinforcers", Value::Numbers(o.range(p, j, 256, points)));
        x.set("acquisitionSteps", o.mean(p, j, 384));
        x.set("resistance", o.mean(p, j, 385));
        const double r = o.mean(p, j, 386);
        x.set("reinforcersPerResponse", r > 0 ? o.mean(p, j, 387) / r : 0.0);
        runs.push(x);
      }
      return Value::object().set("training", training).set("steps", Value::Numbers(steps)).set("runs", runs);
    };
    return m;
  }();
  return &model;
}

}  // namespace syn::models
