// Vygotsky's Zone of Proximal Development in the space.
//
// The learner is the pole L; a task of difficulty b for a learner of ability θ is the point of the
// cognitive channel at logit θ − b, so its chance 1 + ½·T·L is σ(θ − b), the chance of success
// alone. A more knowledgeable other's support s is the dilation T' = shift(T, s); the zones are
// caps about the learner: independent if T is in the cap of chance p_hi, proximal if not but T'
// is in the cap of p_lo, beyond otherwise. Learning from a task is η·P_assisted·(1 − P_alone).
// Plans: 0 the curves over difficulty; 1 Figure 3.2, tasks on the sphere around the learner (at
// angle φ and azimuth α) and their stereographic image (concentric circles); 2 tutoring policies
// over a term, abilities moved by dilations.
#include <cmath>

#include "syn/models/models.hpp"

// The agent program, in WGSL.
constexpr const char* zpd_agent = R"wgsl(
  let pHi = P(0u);
  let pLo = P(1u);
  let scaffold = P(2u);
  let rate = P(3u);
  let kind = J(0u);
  if (kind < f32(0.5)) {
    let task = atLogit(P(4u) - J(1u), kA);
    let helped = shift(task, scaffold);
    let alone = chance(task);
    let assisted = chance(helped);
    let independent = inside(task, chanceCap(pHi));
    let proximal = !independent && inside(helped, chanceCap(pLo));
    stat(0u, alone);
    stat(1u, assisted);
    stat(2u, rate * assisted * (f32(1) - alone));
    stat(3u, select(select(f32(2), f32(1), proximal), f32(0), independent));
  } else if (kind < f32(1.5)) {
    let task = rotate(rotate(pole(), kN, kA, J(1u)), kA, kB, J(2u));
    let helped = shift(task, scaffold);
    let t = unit5(task);
    stat(0u, t.x[1] / (f32(1) + t.x[0]));
    stat(1u, t.x[2] / (f32(1) + t.x[0]));
    let independent = inside(task, chanceCap(pHi));
    let proximal = !independent && inside(helped, chanceCap(pLo));
    stat(3u, select(select(f32(2), f32(1), proximal), f32(0), independent));
  } else {
    let policy = J(1u);
    let sessions = u32(P(5u));
    var ability = sampleLevel(U(0u), P(6u), P(7u), kA);
    let start = logit(ability);
    for (var s = 0u; s < sessions; s = s + 1u) {
      let theta = logit(ability);
      let estimate = theta + P(8u) * kLogisticUnit * logisticOf(U(1u + 2u * s));
      let classLevel = P(9u) + P(10u) * f32(s);
      var b = classLevel;
      if (policy < f32(0.5)) { b = theta - f32(2.5); }
      else if (policy > f32(2.5)) { b = estimate + P(11u); }
      else if (policy > f32(1.5)) { b = theta + f32(5); }
      let task = atLogit(theta - b, kA);
      let alone = chance(task);
      let faded = select(scaffold, scaffold * (f32(0.4) + f32(0.6) * (f32(1) - alone)), P(12u) > f32(0.5));
      let assisted = chance(shift(task, faded));
      let independent = inside(task, chanceCap(pHi));
      let proximal = !independent && inside(shift(task, scaffold), chanceCap(pLo));
      stat(s, theta);
      quant(s, chance(atLogit(f32(0.5) * theta, kA)));
      stat(64u + select(select(2u, 1u, proximal), 0u, independent) * 64u + s, f32(1));
      let gain = rate * assisted * (f32(1) - alone) * (f32(1) + f32(0.2) * kLogisticUnit * logisticOf(U(2u + 2u * s)));
      ability = shift(ability, gain);
    }
    stat(256u, logit(ability) - start);
  }
)wgsl";

namespace syn::models {

namespace {
// The ZPD band for θ = 0: the cap of chance p holds the tasks with Δ ≤ 2·ln tan(ρ/2), cos ρ = 2p − 1.
std::pair<double, double> band(double pHi, double pLo, double scaffold) {
  const double rhoHi = std::acos(2.0 * pHi - 1.0), rhoLo = std::acos(2.0 * pLo - 1.0);
  return {2.0 * std::log(std::tan(rhoHi / 2.0)), scaffold + 2.0 * std::log(std::tan(rhoLo / 2.0))};
}
}  // namespace

const kernel::Model* zpdModel() {
  static const kernel::Model model = [] {
    kernel::Model m;
    m.name = "zpd";
    m.title = "The Zone of Proximal Development: tasks as points, scaffolds as dilations, zones as caps";
    m.maxStats = 257;
    m.agentSource = zpd_agent;
    m.compile = [](const Value& spec) {
      const Value& c = spec.sub("zpd");
      const double pHi = c.num("independent", 0.8), pLo = c.num("assisted", 0.5), s = c.num("scaffold", 1.5), theta = c.num("theta", 0.0);
      const auto b = band(pHi, pLo, s);
      const int sessions = std::min(64, c.integer("sessions", 30));
      const std::vector<double> params = {pHi, pLo, s, c.num("learningRate", 0.35), theta, double(sessions), c.num("abilityMean", 0.0),
                                          c.num("abilitySd", 1.0), c.num("assessmentNoise", 0.4), c.num("classDifficulty", 0.5),
                                          c.num("curriculumPace", 0.08), 0.5 * (b.first + b.second), c.flag("fade", true) ? 1.0 : 0.0};
      std::vector<std::vector<double>> curve, map, tutor;
      for (double d : grid(theta - 5, theta + 7, 120)) curve.push_back({0, d, 0});  // no point exactly on a zone boundary
      const int rings = c.integer("rings", 22), per = c.integer("perRing", 24);
      const double rMin = std::exp(-4.0 / 2.0), rMax = std::exp(3.5 / 2.0);
      for (int r = 0; r < rings; ++r) {
        const double delta = 2.0 * std::log(rMin + (rMax - rMin) * r / (rings - 1));
        for (int k = 0; k < per; ++k) map.push_back({1, 2.0 * std::atan(std::exp(delta / 2.0)), 2.0 * M_PI * (k + 0.5 * (r % 2)) / per});
      }
      for (int policy = 0; policy < 4; ++policy) tutor.push_back({2, double(policy), 0});
      return std::vector<Plan>{makePlan("zpd", 1, 4, params, curve, seedOf(spec)), makePlan("zpd", 1, 4, params, map, seedOf(spec)),
                               makePlan("zpd", static_cast<std::uint32_t>(c.integer("learners", 300)), 257, params, tutor, seedOf(spec),
                                        static_cast<std::uint32_t>(sessions))};
    };
    m.summarise = [](const Value& spec, const std::vector<Plan>& plans, const std::vector<Output>& out) {
      const Value& c = spec.sub("zpd");
      const Plan& p = plans[0];
      Value r = Value::object();
      std::vector<double> difficulty;
      for (std::uint32_t j = 0; j < p.jobs; ++j) difficulty.push_back(p.jobRows[j * p.jobStride + 1]);
      r.set("difficulty", Value::Numbers(difficulty)).set("alone", Value::Numbers(column(p, out[0], 0)));
      r.set("assisted", Value::Numbers(column(p, out[0], 1))).set("gain", Value::Numbers(column(p, out[0], 2)));
      r.set("zone", Value::Numbers(column(p, out[0], 3)));
      const auto b = band(p.params[0], p.params[1], p.params[2]);
      r.set("bandLow", p.params[4] + b.first).set("bandHigh", p.params[4] + b.second);
      const Plan& q = plans[1];
      Value mx = Value::array(), my = Value::array();
      for (int zone = 0; zone < 3; ++zone) {
        std::vector<double> xs, ys;
        for (std::uint32_t j = 0; j < q.jobs; ++j)
          if (std::lround(out[1].mean(q, j, 3)) == zone) {
            xs.push_back(out[1].mean(q, j, 0));
            ys.push_back(out[1].mean(q, j, 1));
          }
        mx.push(Value::Numbers(xs));
        my.push(Value::Numbers(ys));
      }
      r.set("mapX", mx).set("mapY", my).set("mapInner", std::exp(b.first / 2.0)).set("mapOuter", std::exp(b.second / 2.0));
      const Plan& t = plans[2];
      const Output& ot = out[2];
      const int sessions = static_cast<int>(t.params[5]);
      const char* policies[] = {"too easy (review)", "one level for the whole class", "too hard", "targeted at each learner's ZPD"};
      Value runs = Value::array();
      for (std::uint32_t j = 0; j < t.jobs; ++j) {
        std::vector<double> q10, q90;
        auto back = [](double v) { return 2.0 * std::log(v / (1.0 - v)); };
        for (int s = 0; s < sessions; ++s) {
          q10.push_back(back(ot.quantiles[(j * t.quantiles + s) * 3]));
          q90.push_back(back(ot.quantiles[(j * t.quantiles + s) * 3 + 2]));
        }
        Value zones = Value::array();
        for (int z = 0; z < 3; ++z) zones.push(Value::Numbers(ot.range(t, j, 64 + z * 64, sessions)));
        runs.push(Value::object()
                      .set("policy", policies[j])
                      .set("mean", Value::Numbers(ot.range(t, j, 0, sessions)))
                      .set("q10", Value::Numbers(q10))
                      .set("q90", Value::Numbers(q90))
                      .set("zoneShare", zones)
                      .set("gain", ot.mean(t, j, 256)));
      }
      r.set("tutoring", runs);
      (void)c;
      return r;
    };
    return m;
  }();
  return &model;
}

}  // namespace syn::models
