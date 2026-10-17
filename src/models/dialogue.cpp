// Peer dialogue and cultural capital (social constructivism, Bourdieu) in the space.
//
// An agent is a whole class of 32 learners. Each learner's understanding is a point of the
// cognitive channel and their cultural capital (the match of home and school culture) a point of
// the social channel. Every session the teacher's instruction raises understanding by a mixture
// taxed by the capital mismatch unless teaching is inclusive; then learners work in groups (random,
// mixed-ability or same-ability) and move towards more knowledgeable peers, while explaining
// deepens the explainer's own understanding — all mixtures of their points. Jobs: grouping ×
// inclusivity; agents: classes.
#include "syn/models/models.hpp"

// The agent program, in WGSL.
constexpr const char* dialogue_agent = R"wgsl(
  let n = 32u;
  let size = u32(P(0u));
  let sessions = u32(P(1u));
  let instruction = P(2u);
  let peer = P(3u);
  let articulation = P(4u);
  let penalty = P(5u);
  let grouping = J(0u);
  let inclusion = clamp(J(1u), f32(0), f32(1));
  var u : array<V5, 32>;
  var capital : array<V5, 32>;
  var high : array<f32, 32>;
  var sorted : array<f32, 32>;
  for (var i = 0u; i < n; i = i + 1u) {
    capital[i] = sampleShare(U(2u * i), P(8u), P(9u), kC);
    u[i] = atChance(P(6u) + P(7u) * chance(capital[i]) + f32(0.08) * kLogisticUnit * logisticOf(U(2u * i + 1u)), kA);
    sorted[i] = chance(capital[i]);
  }
  for (var i = 1u; i < n; i = i + 1u) {
    let key = sorted[i];
    var j = i;
    while (j > 0u && sorted[j - 1u] > key) { sorted[j] = sorted[j - 1u]; j = j - 1u; }
    sorted[j] = key;
  }
  let median = f32(0.5) * (sorted[15] + sorted[16]);
  var nHigh = f32(0);
  for (var i = 0u; i < n; i = i + 1u) {
    high[i] = select(f32(0), f32(1), chance(capital[i]) > median);
    nHigh = nHigh + high[i];
  }
  var order : array<u32, 32>;
  var group : array<u32, 32>;
  for (var s = 0u; s <= sessions; s = s + 1u) {
    if (s > 0u) {
      var before = u;
      for (var i = 0u; i < n; i = i + 1u) {
        let access = max(f32(0), f32(1) - penalty * (f32(1) - chance(capital[i])) * (f32(1) - inclusion));
        let noise = f32(1) + f32(0.3) * kLogisticUnit * logisticOf(U(100u + s * 100u + i));
        u[i] = nudge(u[i], instruction * access * (f32(1) - chance(before[i])) * noise);
      }
      for (var k = 0u; k < n; k = k + 1u) { order[k] = k; }
      for (var k = n - 1u; k > 0u; k = k - 1u) {
        let r = min(k, u32(floor(U(100u + s * 100u + 40u + k) * f32(k + 1u))));
        let tmp = order[k];
        order[k] = order[r];
        order[r] = tmp;
      }
      if (grouping > f32(1.5)) {
        for (var a = 1u; a < n; a = a + 1u) {
          let key = order[a];
          var b = a;
          while (b > 0u && chance(before[order[b - 1u]]) < chance(before[key])) { order[b] = order[b - 1u]; b = b - 1u; }
          order[b] = key;
        }
      }
      let groups = max(1u, (n + size - 1u) / size);
      for (var k = 0u; k < n; k = k + 1u) {
        group[order[k]] = select(k % groups, min(groups - 1u, k / size), grouping > f32(2.5));
      }
      if (grouping > f32(0.5)) {
        for (var i = 0u; i < n; i = i + 1u) {
          var learnFrom = f32(0);
          var explainTo = f32(0);
          var others = f32(0);
          for (var j = 0u; j < n; j = j + 1u) {
            if (j != i && group[j] == group[i]) {
              learnFrom = learnFrom + max(f32(0), chance(before[j]) - chance(before[i]));
              explainTo = explainTo + max(f32(0), chance(before[i]) - chance(before[j]));
              others = others + f32(1);
            }
          }
          if (others > f32(0)) {
            u[i] = nudge(u[i], peer * learnFrom / others + articulation * explainTo / others * (f32(1) - chance(before[i])));
          }
        }
      }
    }
    var total = f32(0);
    var hi = f32(0);
    var sq = f32(0);
    for (var i = 0u; i < n; i = i + 1u) {
      let x = chance(u[i]);
      total = total + x;
      sq = sq + x * x;
      hi = hi + x * high[i];
    }
    let mean = total / f32(n);
    stat(s, mean);
    stat(32u + s, select(f32(0), hi / nHigh, nHigh > f32(0)));
    stat(64u + s, select(f32(0), (total - hi) / (f32(n) - nHigh), nHigh < f32(n)));
    stat(96u + s, sqrt(max(f32(0), (sq - f32(n) * mean * mean) / f32(n - 1u))));
  }
)wgsl";

namespace syn::models {

const kernel::Model* dialogueModel() {
  static const kernel::Model model = [] {
    kernel::Model m;
    m.name = "dialogue";
    m.title = "Peer dialogue and cultural capital: classes of learners moved by mixtures";
    m.maxStats = 128;
    m.agentSource = dialogue_agent;
    m.compile = [](const Value& spec) {
      const Value& c = spec.sub("dialogue");
      const int sessions = std::min(31, c.integer("sessions", 20));
      const std::vector<double> params = {double(std::max(2, c.integer("groupSize", 4))), double(sessions), c.num("instructionRate", 0.06),
                                          c.num("peerRate", 0.25), c.num("articulationRate", 0.08), c.num("capitalPenalty", 0.6),
                                          c.num("priorBase", 0.15), c.num("priorCapital", 0.2), c.num("capitalMean", 0.55), c.num("capitalSd", 0.25)};
      std::vector<std::vector<double>> rows;
      for (int g = 0; g < 4; ++g)
        for (double inclusivity : {0.0, c.num("inclusivity", 1.0)}) rows.push_back({double(g), inclusivity});
      return std::vector<Plan>{makePlan("dialogue", static_cast<std::uint32_t>(c.integer("classes", 50)), 128, params, rows, seedOf(spec))};
    };
    m.summarise = [](const Value&, const std::vector<Plan>& plans, const std::vector<Output>& out) {
      const Plan& p = plans[0];
      const Output& o = out[0];
      const int steps = static_cast<int>(p.params[1]) + 1;
      const char* names[] = {"individual work (no dialogue)", "random groups", "mixed-ability groups", "same-ability groups"};
      Value runs = Value::array();
      for (std::uint32_t j = 0; j < p.jobs; ++j) {
        const auto mean = o.range(p, j, 0, steps), high = o.range(p, j, 32, steps), low = o.range(p, j, 64, steps);
        runs.push(Value::object()
                      .set("grouping", names[static_cast<int>(p.jobRows[j * p.jobStride])])
                      .set("inclusivity", p.jobRows[j * p.jobStride + 1])
                      .set("mean", Value::Numbers(mean))
                      .set("high", Value::Numbers(high))
                      .set("low", Value::Numbers(low))
                      .set("spread", Value::Numbers(o.range(p, j, 96, steps)))
                      .set("finalMean", mean.back())
                      .set("finalGap", high.back() - low.back()));
      }
      return Value::object().set("runs", runs);
    };
    return m;
  }();
  return &model;
}

}  // namespace syn::models
