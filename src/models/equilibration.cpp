// Piaget's equilibration (constructivism) in the space.
//
// A child's experiences of animals — [size, ear pointiness, sound] — are lifted onto S³ by the
// conformal stereographic map. A schema is a prototype point and its assimilation cap
// S = p̂ + cos ρ·e. An experience inside the cap of its nearest schema (largest X·P) that the world
// does not contradict is assimilated: the rotor exp(−ηθ/2·B̂) turns the prototype a fraction η
// of the way towards it. Otherwise it is a conflict: disequilibrium D ← ρD + (1 − ρ) builds up,
// unless the learner avoids the discomfort; past a threshold the learner accommodates, building
// a new schema at the centre of the conflicting experiences (the direction of the sum of their
// null vectors), named by the feedback received. Jobs: feedback and avoidance conditions;
// agents: children; accuracy on a fixed test set of 100 dogs and cats after each observation.
#include <cmath>

#include "syn/models/models.hpp"
#include "syn/kernel/agent.hpp"

SYN_AGENT(equilibration_agent,
  let n = u32(P(0u));
  let catsFrom = u32(P(1u));
  let catShare = P(2u);
  let noise = P(3u);
  let scale = P(4u);
  let radius = P(5u);
  let rate = P(6u);
  let persist = P(7u);
  let threshold = P(8u);
  let feedback = J(0u);
  let avoidance = J(1u);
  ARR(test, V5, 100);
  for (var k = 0u; k < 100u; k = k + 1u) { test[k] = lift(P(16u + 4u * k), P(17u + 4u * k), P(18u + 4u * k), scale); }
  ARR(proto, V5, 8);
  ARR(label, f32, 8);
  proto[0] = lift(f32(1), f32(0), f32(0), scale);
  label[0] = f32(0);
  var count = 1u;
  ARR(sums, V5, 3);
  ARR(votes, f32, 3);
  for (var k = 0u; k < 3u; k = k + 1u) { sums[k] = v5(f32(0), f32(0), f32(0), f32(0), f32(0)); }
  var d = f32(0);
  for (var t = 0u; t < n; t = t + 1u) {
    let isCat = t >= catsFrom && U(4u * t) < catShare;
    let named = U(4u * t + 1u) < feedback;
    let kind = select(f32(0), f32(1), isCat);
    let cx = select(f32(1), f32(0.3), isCat) + noise * kLogisticUnit * logisticOf(U(1000u + 3u * t));
    let cy = kind + noise * kLogisticUnit * logisticOf(U(1001u + 3u * t));
    let cz = kind + noise * kLogisticUnit * logisticOf(U(1002u + 3u * t));
    let X = lift(cx, cy, cz, scale);
    var chosenSchema = 0u;
    var best = f32(-1e30);
    for (var i = 0u; i < count; i = i + 1u) {
      let dd = dot5(X, proto[i]);
      if (dd > best) { best = dd; chosenSchema = i; }
    }
    var ok = inside(X, cap(proto[chosenSchema], radius)) && (!named || label[chosenSchema] == kind);
    if (!ok && named) {
      var best2 = f32(-1e30);
      for (var i = 0u; i < count; i = i + 1u) {
        let dd = dot5(X, proto[i]);
        if (label[i] == kind && inside(X, cap(proto[i], radius)) && dd > best2) { best2 = dd; chosenSchema = i; ok = true; }
      }
    }
    if (ok) {
      proto[chosenSchema] = towards(proto[chosenSchema], X, rate);
      d = d * persist;
      stat(512u, f32(1));
    } else if (U(4u * t + 2u) < avoidance) {
      d = d * persist;
      stat(514u, f32(1));
    } else {
      d = persist * d + (f32(1) - persist);
      stat(515u, f32(1));
      let slot = select(0u, 1u + u32(kind), named);
      sums[slot] = add5(sums[slot], X);
      votes[slot] = votes[slot] + f32(1);
      if (d >= threshold) {
        var newLabel = f32(1000);
        for (var i = 0u; i < count; i = i + 1u) { newLabel = max(newLabel, label[i] + f32(1)); }
        var members = sums[0];
        if (votes[1] > f32(0) || votes[2] > f32(0)) {
          let chosen = select(1u, 2u, votes[2] > votes[1]);
          newLabel = f32(chosen - 1u);
          members = add5(members, sums[chosen]);
        }
        if (count < 8u) {
          proto[count] = pointOf(members);
          label[count] = newLabel;
          count = count + 1u;
        }
        for (var k = 0u; k < 3u; k = k + 1u) {
          sums[k] = v5(f32(0), f32(0), f32(0), f32(0), f32(0));
          votes[k] = f32(0);
        }
        d = d * f32(0.3);
        stat(513u, f32(1));
      }
    }
    var correct = f32(0);
    for (var k = 0u; k < 100u; k = k + 1u) {
      var nearest = 0u;
      var top = f32(-1e30);
      for (var i = 0u; i < count; i = i + 1u) {
        let dd = dot5(test[k], proto[i]);
        if (dd > top) { top = dd; nearest = i; }
      }
      if (label[nearest] == P(19u + 4u * k)) { correct = correct + f32(1); }
    }
    stat(t, d);
    stat(128u + t, correct / f32(100));
    stat(256u + t, f32(count));
    stat(384u + t, f32(named));
  }
)

namespace syn::models {

const kernel::Model* equilibrationModel() {
  static const kernel::Model model = [] {
    kernel::Model m;
    m.name = "equilibration";
    m.title = "Piaget's equilibration: assimilation by rotors, accommodation at the centre of conflicts";
    m.maxStats = 516;
    SYN_USE_AGENT(m, equilibration_agent);
    m.compile = [](const Value& spec) {
      const Value& c = spec.sub("equilibration");
      const int n = std::min(128, c.integer("observations", 120));
      const double noise = c.num("noise", 0.35);
      std::vector<double> params = {double(n), double(c.integer("catsFrom", 20)), c.num("catShare", 0.5), noise, c.num("featureScale", 2.0),
                                    c.num("radius", 1.25), c.num("learningRate", 0.08), c.num("persistence", 0.7), c.num("threshold", 0.45)};
      params.resize(16, 0.0);
      // A fixed test set of 50 dogs and 50 cats.
      const std::uint32_t testSeed = seedOf(spec) ^ 0x7e57u;
      for (std::uint32_t i = 0; i < 100; ++i) {
        const bool cat = i % 2 == 1;
        params.push_back((cat ? 0.3 : 1.0) + noise * logisticDraw(testSeed, i, 0));
        params.push_back((cat ? 1.0 : 0.0) + noise * logisticDraw(testSeed, i, 1));
        params.push_back((cat ? 1.0 : 0.0) + noise * logisticDraw(testSeed, i, 2));
        params.push_back(cat ? 1.0 : 0.0);
      }
      const double feedback = c.num("feedback", 0.3);
      const std::vector<std::vector<double>> rows = {{0.0, 0.0}, {feedback, 0.0}, {1.0, 0.0}, {feedback, c.num("avoidance", 0.7)}};
      return std::vector<Plan>{makePlan("equilibration", static_cast<std::uint32_t>(c.integer("learners", 200)), 516, params, rows, seedOf(spec))};
    };
    m.summarise = [](const Value& spec, const std::vector<Plan>& plans, const std::vector<Output>& out) {
      const Plan& p = plans[0];
      const Output& o = out[0];
      const int n = static_cast<int>(p.params[0]);
      const char* labels[] = {"no one names the animals", "adults sometimes name them", "adults always name them", "named, but discomfort is avoided"};
      Value runs = Value::array();
      for (std::uint32_t j = 0; j < p.jobs; ++j) {
        const auto accuracy = o.range(p, j, 128, n);
        runs.push(Value::object()
                      .set("label", labels[j])
                      .set("feedback", p.jobRows[j * p.jobStride])
                      .set("avoidance", p.jobRows[j * p.jobStride + 1])
                      .set("accuracy", Value::Numbers(accuracy))
                      .set("disequilibrium", Value::Numbers(o.range(p, j, 0, n)))
                      .set("schemas", Value::Numbers(o.range(p, j, 256, n)))
                      .set("finalAccuracy", accuracy.back())
                      .set("assimilations", o.mean(p, j, 512))
                      .set("accommodations", o.mean(p, j, 513))
                      .set("avoidances", o.mean(p, j, 514))
                      .set("conflicts", o.mean(p, j, 515)));
      }
      return Value::object().set("runs", runs).set("catsFrom", spec.sub("equilibration").integer("catsFrom", 20));
    };
    return m;
  }();
  return &model;
}

}  // namespace syn::models
