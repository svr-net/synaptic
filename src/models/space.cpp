// The space itself (core page): scalars as the miniature basis of the elliptic CGA.
//
// Plan 0, one job per logit x on a grid: the point of logit x and what the algebra does to it —
// its chance against σ(x), a dilation (adds d to the logit), a mixture towards the learner (moves
// the chance linearly), a cap (a threshold), its angle from the pole, the same level in another
// channel, the logit read back, and a rotor between channels (keeps the chance).
// Plan 1, one job, many agents: random points sampled uniformly in chance; their logits are
// logistic (the space's own coordinate), histogrammed into 32 bins.
#include "syn/models/models.hpp"

// The agent program, in WGSL.
constexpr const char* space_agent = R"wgsl(
  if (P(0u) < f32(0.5)) {
    let x = J(0u);
    let X = atLogit(x, kA);
    stat(0u, chance(X));
    stat(1u, f32(1) / (f32(1) + exp(-x)));
    stat(2u, chance(shift(X, P(1u))));
    stat(3u, f32(1) / (f32(1) + exp(-(x + P(1u)))));
    stat(4u, chance(mixture(X, pole(), P(2u))));
    stat(5u, select(f32(0), f32(1), inside(X, chanceCap(P(3u)))));
    stat(6u, acos(clamp(unit5(X).x[0], f32(-1), f32(1))));
    stat(7u, abs(dot5(X, X)));
    stat(8u, similarity(X, atLogit(x, kB)));
    stat(9u, logit(X));
    stat(10u, abs(chance(rotate(X, 1, 2, P(4u))) - chance(X)));
  } else {
    let X = sampleLevel(U(0u), P(5u), P(6u), kA);
    let x = logit(X);
    stat(0u, x);
    stat(1u, x * x);
    stat(2u, chance(X));
    stat(3u, abs(dot5(X, X)));
    let b = u32(clamp((x + f32(6)) / f32(12) * f32(32), f32(0), f32(31)));
    stat(4u + b, f32(1));
    quant(0u, chance(X));
  }
)wgsl";

namespace syn::models {

const kernel::Model* spaceModel() {
  static const kernel::Model model = [] {
    kernel::Model m;
    m.name = "space";
    m.title = "the space: logits, dilations, mixtures, caps and random points in Cl(4, 1)";
    m.maxStats = 36;
    m.agentSource = space_agent;
    m.compile = [](const Value& spec) {
      const Value& c = spec.sub("space");
      const double d = c.num("shift", 1.5), t = c.num("mixture", 0.3), p = c.num("cap", 0.7), alpha = c.num("rotation", 0.8);
      const double mean = c.num("mean", 0.0), sd = c.num("sd", 1.0);
      const std::vector<double> params = {0, d, t, p, alpha, mean, sd};
      std::vector<std::vector<double>> rows;
      for (double x : grid(-6.0, 6.0, c.integer("points", 121))) rows.push_back({x});
      std::vector<double> random = params;
      random[0] = 1;
      return std::vector<Plan>{makePlan("space", 1, 11, params, rows, seedOf(spec)),
                               makePlan("space", static_cast<std::uint32_t>(c.integer("samples", 20000)), 36, random, {{0.0}}, seedOf(spec), 1)};
    };
    m.summarise = [](const Value& spec, const std::vector<Plan>& plans, const std::vector<Output>& out) {
      const Plan& p = plans[0];
      const Output& o = out[0];
      Value r = Value::object();
      std::vector<double> x;
      for (std::uint32_t j = 0; j < p.jobs; ++j) x.push_back(p.jobRows[j]);
      r.set("x", Value::Numbers(x));
      const char* names[] = {"chance", "logistic", "shifted", "logisticShifted", "mixed", "insideCap", "angle", "nullError", "crossChannel", "logitBack", "rotorChange"};
      for (std::uint32_t k = 0; k < 11; ++k) r.set(names[k], Value::Numbers(column(p, o, k)));
      double chanceErr = 0, shiftErr = 0, nullErr = 0, logitErr = 0, rotorErr = 0, mixErr = 0;
      const double t = p.params[2];
      for (std::uint32_t j = 0; j < p.jobs; ++j) {
        chanceErr = std::max(chanceErr, std::fabs(o.mean(p, j, 0) - o.mean(p, j, 1)));
        shiftErr = std::max(shiftErr, std::fabs(o.mean(p, j, 2) - o.mean(p, j, 3)));
        mixErr = std::max(mixErr, std::fabs(o.mean(p, j, 4) - ((1 - t) * o.mean(p, j, 0) + t)));
        nullErr = std::max(nullErr, o.mean(p, j, 7));
        logitErr = std::max(logitErr, std::fabs(o.mean(p, j, 9) - x[j]));
        rotorErr = std::max(rotorErr, o.mean(p, j, 10));
      }
      Value checks = Value::object();
      checks.set("chanceIsLogistic", chanceErr).set("dilationAddsLogits", shiftErr).set("mixtureIsLinear", mixErr);
      checks.set("pointsAreNull", nullErr).set("logitReadBack", logitErr).set("rotorKeepsChance", rotorErr);
      r.set("checks", checks);
      r.set("params", Value::object().set("shift", p.params[1]).set("mixture", t).set("cap", p.params[3]).set("rotation", p.params[4]));
      const Plan& q = plans[1];
      const Output& oq = out[1];
      Value rnd = Value::object();
      const double mean = oq.mean(q, 0, 0), m2 = oq.mean(q, 0, 1);
      rnd.set("samples", static_cast<double>(q.agents));
      rnd.set("meanLogit", mean).set("sdLogit", std::sqrt(std::max(0.0, m2 - mean * mean))).set("meanChance", oq.mean(q, 0, 2));
      rnd.set("nullError", oq.mean(q, 0, 3));
      std::vector<double> bins = oq.range(q, 0, 4, 32), centres;
      for (int b = 0; b < 32; ++b) centres.push_back(-6.0 + 12.0 * (b + 0.5) / 32);
      rnd.set("binCentres", Value::Numbers(centres)).set("binShare", Value::Numbers(bins));
      rnd.set("chanceQuantiles", Value::Numbers(std::vector<double>(oq.quantiles.begin(), oq.quantiles.end())));
      r.set("random", rnd);
      (void)spec;
      return r;
    };
    return m;
  }();
  return &model;
}

}  // namespace syn::models
