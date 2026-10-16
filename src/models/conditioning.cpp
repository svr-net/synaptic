// Classical conditioning (Pavlov, Watson, Kamin) in the space.
//
// An associative strength V ∈ [0, 1] (λ = 1) is a point of the cognitive channel; the prediction
// of the US by the stimuli present is the sum of their readings, and the Rescorla–Wagner trial
//   ΔV = α·β·(λ − ΣV)
// is a mixture of each present stimulus' point towards the learner (or towards the antipode when
// the error is negative): nudge(V, α·β·error). Jobs:
//   0  Pavlov: bell + food, then the bell alone; with the closed form λ(1 − (1 − αβ)ⁿ)
//   1  Kamin blocking: light → food, then light + tone → food
//   2  the blocking control: light + tone → food only
//   3  Watson: a white rat paired with a loud noise; fear generalises to stimuli by the similarity
//      of their points, 1 + ½·X·Y = cos²(θ/2), θ the angle between the stimuli on the sphere.
#include "syn/models/models.hpp"
#include "syn/kernel/agent.hpp"

SYN_AGENT(conditioning_agent,
  let kind = J(0u);
  let alpha = J(1u);
  let n1 = u32(J(2u));
  let n2 = u32(J(3u));
  let betaUs = J(4u);
  let betaNoUs = J(5u);
  var light = atChance(f32(0), kA);
  var tone = atChance(f32(0), kA);
  if (kind < f32(0.5)) {
    for (var t = 0u; t < n1 + n2; t = t + 1u) {
      let us = t < n1;
      let beta = select(betaNoUs, betaUs, us);
      let err = select(f32(0), f32(1), us) - chance(light);
      light = nudge(light, alpha * beta * err);
      stat(t, chance(light));
      let acquired = f32(1) - pow(f32(1) - alpha * betaUs, f32(n1));
      let closed = select(acquired * pow(f32(1) - alpha * betaNoUs, f32(t + 1u - n1)), f32(1) - pow(f32(1) - alpha * betaUs, f32(t + 1u)), us);
      stat(64u + t, closed);
    }
  } else if (kind < f32(2.5)) {
    let pre = select(n1, 0u, kind > f32(1.5));
    for (var t = 0u; t < pre + n2; t = t + 1u) {
      let compound = t >= pre;
      let predicted = chance(light) + select(f32(0), chance(tone), compound);
      let err = f32(1) - predicted;
      light = nudge(light, alpha * betaUs * err);
      if (compound) { tone = nudge(tone, alpha * betaUs * err); }
      stat(t, chance(light));
      stat(64u + t, chance(tone));
    }
  } else {
    var fear = atChance(f32(0), kA);
    for (var t = 0u; t < n1; t = t + 1u) {
      fear = nudge(fear, alpha * betaUs * (f32(1) - chance(fear)));
    }
    stat(16u, chance(fear));
    let rat = equator(kC);
    for (var k = 0u; k < 6u; k = k + 1u) {
      let probe = rotate(rotate(rat, kC, kA, P(2u * k)), kA, kB, P(2u * k + 1u));
      let sim = similarity(rat, probe);
      stat(k, chance(fear) * pow(sim, f32(1) / P(12u)));
      stat(8u + k, f32(2) * acos(sqrt(clamp(sim, f32(0), f32(1)))));
    }
  }
)

namespace syn::models {

const kernel::Model* conditioningModel() {
  static const kernel::Model model = [] {
    kernel::Model m;
    m.name = "conditioning";
    m.title = "Rescorla–Wagner conditioning: Pavlov, Kamin blocking, Watson's generalised fear";
    m.maxStats = 128;
    SYN_USE_AGENT(m, conditioning_agent);
    m.compile = [](const Value& spec) {
      const Value& c = spec.sub("conditioning");
      const double salience = c.num("salience", 0.3);
      const int acq = std::min(32, c.integer("acquisition", 30)), ext = std::min(32, c.integer("extinction", 30));
      const int pre = std::min(32, c.integer("pretraining", 20)), comp = std::min(32, c.integer("compound", 20));
      // Stimuli by their angle from the white rat and their azimuth (furry, white, animate...).
      const std::vector<double> probes = {0.0, 0.0, 0.4, 0.3, 0.8, 1.9, 1.0, 3.1, 1.4, 4.4, 3.0, 5.5, c.num("width", 1.0)};
      const std::vector<std::vector<double>> rows = {
          {0, salience, double(acq), double(ext), 0.5, 0.3},
          {1, 0.3, double(pre), double(comp), 0.5, 0.3},
          {2, 0.3, double(pre), double(comp), 0.5, 0.3},
          {3, 0.4, double(c.integer("pairings", 7)), 0, 0.8, 0.3},
      };
      return std::vector<Plan>{makePlan("conditioning", 1, 128, probes, rows, seedOf(spec))};
    };
    m.summarise = [](const Value&, const std::vector<Plan>& plans, const std::vector<Output>& out) {
      const Plan& p = plans[0];
      const Output& o = out[0];
      const auto row = [&](std::uint32_t j, std::uint32_t k) { return static_cast<int>(p.jobRows[j * p.jobStride + k]); };
      Value r = Value::object();
      const int acq = row(0, 2), ext = row(0, 3);
      r.set("pavlov", Value::Numbers(o.range(p, 0, 0, acq + ext)));
      r.set("closedForm", Value::Numbers(o.range(p, 0, 64, acq + ext)));
      r.set("acquisitionTrials", acq);
      const int pre = row(1, 2), comp = row(1, 3);
      Value b = Value::object();
      b.set("light", Value::Numbers(o.range(p, 1, 0, pre + comp)));
      b.set("tone", Value::Numbers(o.range(p, 1, 64, pre + comp)));
      b.set("controlTone", Value::Numbers(o.range(p, 2, 64, comp)));
      b.set("blockedTone", o.mean(p, 1, 64 + pre + comp - 1));
      b.set("controlToneFinal", o.mean(p, 2, 64 + comp - 1));
      b.set("compoundStart", pre);
      r.set("blocking", b);
      const char* names[] = {"white rat", "white rabbit", "dog", "fur coat", "cotton wool", "wooden blocks"};
      Value probes = Value::array();
      for (std::uint32_t k = 0; k < 6; ++k)
        probes.push(Value::object().set("stimulus", names[k]).set("distance", o.mean(p, 3, 8 + k)).set("fear", o.mean(p, 3, k)));
      r.set("fear", probes);
      r.set("fearAcquired", o.mean(p, 3, 16));
      return r;
    };
    return m;
  }();
  return &model;
}

}  // namespace syn::models
