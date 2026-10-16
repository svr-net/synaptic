// Andragogy (Knowles, Bartle) in the space.
//
// Each of Knowles' six assumptions about the learner and each matching design feature is a point
// of the affective channel; 2p − 1 is the N-coordinate of its point (cos φ). Engagement is the
// equator point dilated by the intercept and, for every dimension, by the feature's main effect
// m·d and its interaction w·(2o − 1)(2d − 1) with the learner's assumption: dilations compose by
// adding logits. Bartle's needs-diagnosis cycle: a learner's competency gaps are shares (of a gap
// of 4) that each cycle's objectives lower; learner input and a cooperative climate make the
// diagnosis of the largest gaps more accurate. Plans: engagement for a child, an adolescent and an
// adult as the design moves from pedagogical to andragogical; the needs cycle at three levels of
// input and climate.
#include <string>

#include "syn/models/models.hpp"
#include "syn/kernel/agent.hpp"

SYN_AGENT(andragogy_agent,
  if (J(0u) < f32(0.5)) {
    let learner = u32(J(1u));
    let mixing = J(2u);
    var engage = shift(equator(kB), P(0u));
    for (var k = 0u; k < 6u; k = k + 1u) {
      let o = atChance(P(13u + 6u * learner + k), kB);
      let d = atChance((f32(1) - mixing) * P(31u + k) + mixing * P(37u + k), kB);
      let term = P(1u + k) * chance(d) + P(7u + k) * unit5(o).x[0] * unit5(d).x[0];
      engage = shift(engage, term);
      stat(1u + k, term);
    }
    stat(0u, chance(engage));
  } else {
    let competencies = 8u;
    let cycles = u32(P(43u));
    let objectives = u32(P(44u));
    let input = clamp(J(1u), f32(0), f32(1));
    let noise = P(45u) * (f32(1) - f32(0.6) * input) * (f32(1) - f32(0.4) * input);
    ARR(gap, V5, 8);
    ARR(diagnosed, f32, 8);
    var total = f32(0);
    for (var j = 0u; j < competencies; j = j + 1u) {
      let g = max(f32(0), f32(1) + f32(0.6) * kLogisticUnit * logisticOf(U(j)));
      gap[j] = atChance(g / f32(4), kA);
      total = total + f32(4) * chance(gap[j]);
    }
    stat(8u, total / f32(competencies));
    for (var c = 0u; c < cycles; c = c + 1u) {
      for (var j = 0u; j < competencies; j = j + 1u) {
        diagnosed[j] = f32(4) * chance(gap[j]) + noise * kLogisticUnit * logisticOf(U(100u + c * 16u + j));
      }
      var hits = f32(0);
      ARR(chosen, u32, 8);
      for (var a = 0u; a < objectives; a = a + 1u) {
        var best = 0u;
        var top = f32(-1e30);
        for (var j = 0u; j < competencies; j = j + 1u) {
          var taken = false;
          for (var b = 0u; b < a; b = b + 1u) { if (chosen[b] == j) { taken = true; } }
          if (!taken && diagnosed[j] > top) { top = diagnosed[j]; best = j; }
        }
        chosen[a] = best;
      }
      for (var a = 0u; a < objectives; a = a + 1u) {
        var larger = 0u;
        let mine = chance(gap[chosen[a]]);
        for (var j = 0u; j < competencies; j = j + 1u) { if (chance(gap[j]) > mine) { larger = larger + 1u; } }
        if (larger < objectives) { hits = hits + f32(1); }
      }
      stat(16u + c, hits / f32(objectives));
      for (var a = 0u; a < objectives; a = a + 1u) { gap[chosen[a]] = lower(gap[chosen[a]], P(46u)); }
      var sum = f32(0);
      for (var j = 0u; j < competencies; j = j + 1u) { sum = sum + f32(4) * chance(gap[j]); }
      stat(9u + c, sum / f32(competencies));
    }
  }
)

namespace syn::models {

namespace {
const char* kAssumptions[6] = {"self-directed self-concept", "reservoir of experience", "readiness tied to social roles",
                               "performance-centred orientation", "internal motivation", "need to know why"};
const char* kFeatures[6] = {"learner input", "use of experience", "relevance to roles", "immediate application", "intrinsic appeal (no rewards)",
                            "transparency of purpose"};
const char* kSteps[7] = {"set a cooperative learning climate", "create mechanisms for input", "diagnose learner needs and interests",
                         "formulate learning objectives", "design sequential activities", "execute the design", "evaluate and rediagnose needs"};
}  // namespace

const kernel::Model* andragogyModel() {
  static const kernel::Model model = [] {
    kernel::Model m;
    m.name = "andragogy";
    m.title = "Andragogy: engagement as dilations by assumptions × design; the needs-diagnosis cycle";
    m.maxStats = 24;
    SYN_USE_AGENT(m, andragogy_agent);
    m.compile = [](const Value& spec) {
      const Value& c = spec.sub("andragogy");
      std::vector<double> params = {0.4, 0.05, 0.05, 0.1, 0.05, 0.0, 0.25, 0.45, 0.3, 0.35, 0.35, 0.35, 0.25};
      // Orientations: child, adolescent, adult (13..30); designs: pedagogical (31..36), andragogical (37..42).
      for (double x : {0.15, 0.1, 0.15, 0.2, 0.35, 0.3, 0.45, 0.35, 0.4, 0.45, 0.45, 0.6, 0.85, 0.85, 0.85, 0.85, 0.8, 0.9}) params.push_back(x);
      for (double x : {0.1, 0.1, 0.2, 0.2, 0.2, 0.3, 0.9, 0.9, 0.9, 0.9, 0.85, 0.9}) params.push_back(x);
      const int cycles = std::min(7, c.integer("cycles", 6));
      params.insert(params.end(), {double(cycles), double(std::min(8, c.integer("objectives", 2))), c.num("diagnosisNoise", 1.0), c.num("gainPerObjective", 0.45)});
      std::vector<std::vector<double>> curves, cycle;
      for (int l = 0; l < 3; ++l)
        for (double mix : grid(0, 1, 21)) curves.push_back({0, double(l), mix});
      for (double input : {0.0, 0.5, 1.0}) cycle.push_back({1, input, 0});
      return std::vector<Plan>{makePlan("andragogy", 1, 7, params, curves, seedOf(spec)),
                               makePlan("andragogy", static_cast<std::uint32_t>(c.integer("learners", 300)), 24, params, cycle, seedOf(spec))};
    };
    m.summarise = [](const Value&, const std::vector<Plan>& plans, const std::vector<Output>& out) {
      const Plan& p = plans[0];
      const Output& o = out[0];
      Value r = Value::object();
      r.set("mix", Value::Numbers(grid(0, 1, 21)));
      Value curves = Value::array(), shift = Value::array();
      for (std::uint32_t l = 0; l < 3; ++l) {
        std::vector<double> e, d;
        for (std::uint32_t k = 0; k < 21; ++k) e.push_back(o.mean(p, l * 21 + k, 0));
        for (std::uint32_t k = 0; k < 6; ++k) d.push_back(o.mean(p, l * 21 + 20, 1 + k) - o.mean(p, l * 21, 1 + k));
        curves.push(Value::Numbers(e));
        shift.push(Value::Numbers(d));
      }
      r.set("curves", curves).set("termShift", shift);
      r.set("assumptions", std::vector<std::string>(kAssumptions, kAssumptions + 6));
      r.set("features", std::vector<std::string>(kFeatures, kFeatures + 6));
      r.set("steps", std::vector<std::string>(kSteps, kSteps + 7));
      const Plan& q = plans[1];
      const int cycles = static_cast<int>(q.params[43]);
      Value list = Value::array();
      for (std::uint32_t j = 0; j < q.jobs; ++j)
        list.push(Value::object()
                      .set("input", q.jobRows[j * q.jobStride + 1])
                      .set("gap", Value::Numbers(out[1].range(q, j, 8, cycles + 1)))
                      .set("hitRate", Value::Numbers(out[1].range(q, j, 16, cycles))));
      r.set("cycles", list);
      return r;
    };
    return m;
  }();
  return &model;
}

}  // namespace syn::models
