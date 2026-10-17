// Schemas and spreading activation (cognitivism) in the space.
//
// The activation of every concept of a small schema of library-search knowledge is a point of the
// cognitive channel. Activation spreads along the links: a concept's point is raised to the
// strongest incoming route, decay · weight · activation of the neighbour (the max keeps it in
// [0, 1]). A new concept ("AND operator") presented with the Venn diagram is linked to every
// concept active enough at that moment, with a weight proportional to the activation — and
// stimulating the recall of prior knowledge (Gagné's third event) makes more of them active.
// Recalling AND from a cue is the chance of the equator point dilated by
// gain · (activation of AND − threshold). Jobs: (plain, primed) × the cue concept.
#include "syn/models/models.hpp"

// The agent program, in WGSL.
constexpr const char* schema_agent = R"wgsl(
  let primed = J(0u) > f32(0.5);
  let cue = u32(J(1u));
  var w : array<f32, 121>;
  for (var i = 0u; i < 100u; i = i + 1u) { w[(i / 10u) * 11u + i % 10u] = P(i); }
  var act : array<V5, 11>;
  var degree = f32(0);
  for (var phase = 0u; phase < 2u; phase = phase + 1u) {
    for (var i = 0u; i < 11u; i = i + 1u) { act[i] = atChance(f32(0), kA); }
    if (phase == 0u) {
      act[4] = atChance(f32(1), kA);
      if (primed) {
        act[6] = atChance(f32(1), kA);
        act[1] = atChance(f32(1), kA);
        act[0] = atChance(f32(1), kA);
      }
    } else {
      act[cue] = atChance(f32(1), kA);
    }
    let steps = select(3u, 1u, phase == 0u);
    for (var s = 0u; s < steps; s = s + 1u) {
      var next = act;
      for (var i = 0u; i < 11u; i = i + 1u) {
        var incoming = f32(0);
        for (var j = 0u; j < 11u; j = j + 1u) { incoming = max(incoming, P(102u) * w[j * 11u + i] * chance(act[j])); }
        if (incoming > chance(act[i])) { next[i] = atChance(incoming, kA); }
      }
      act = next;
    }
    if (phase == 0u) {
      for (var j = 0u; j < 10u; j = j + 1u) {
        let a = chance(act[j]);
        if (a >= P(100u)) {
          let wj = clamp(P(101u) * a, f32(0), f32(1));
          w[110u + j] = wj;
          w[j * 11u + 10u] = wj;
          degree = degree + f32(1);
        }
      }
    }
  }
  stat(0u, chance(shift(equator(kA), P(104u) * (chance(act[10]) - P(103u)))));
  for (var j = 0u; j < 10u; j = j + 1u) { stat(1u + j, w[110u + j]); }
  stat(11u, degree);
)wgsl";

namespace syn::models {

namespace {
const char* kConcepts[11] = {"research question", "keyword", "synonym", "search engine", "Venn diagram", "set",
                             "narrow results", "broaden results", "exclude a term", "relevance", "AND operator"};
}  // namespace

const kernel::Model* schemaModel() {
  static const kernel::Model model = [] {
    kernel::Model m;
    m.name = "schema";
    m.title = "Schemas: spreading activation and encoding a new concept with prior knowledge";
    m.maxStats = 12;
    m.agentSource = schema_agent;
    m.compile = [](const Value& spec) {
      const Value& c = spec.sub("schema");
      std::vector<double> params(105, 0.0);
      auto link = [&](int a, int b, double w) { params[a * 10 + b] = params[b * 10 + a] = w; };
      link(0, 1, 0.9);
      link(1, 2, 0.8);
      link(1, 3, 0.7);
      link(3, 9, 0.6);
      link(4, 5, 0.9);
      link(5, 6, 0.5);
      link(5, 7, 0.5);
      link(5, 8, 0.5);
      link(6, 9, 0.6);
      link(7, 2, 0.6);
      params[100] = c.num("linkThreshold", 0.2);
      params[101] = c.num("strength", 0.8);
      params[102] = c.num("decay", 0.6);
      params[103] = c.num("threshold", 0.25);
      params[104] = c.num("gain", 12.0);
      std::vector<std::vector<double>> rows;
      for (double primed : {0.0, 1.0})
        for (int cue = 0; cue < 11; ++cue) rows.push_back({primed, double(cue)});
      return std::vector<Plan>{makePlan("schema", 1, 12, params, rows, seedOf(spec))};
    };
    m.summarise = [](const Value&, const std::vector<Plan>& plans, const std::vector<Output>& out) {
      const Plan& p = plans[0];
      const Output& o = out[0];
      auto network = [&](std::uint32_t first) {
        Value x = Value::object();
        x.set("nodes", std::vector<std::string>(kConcepts, kConcepts + 11));
        Value edges = Value::array();
        for (int i = 0; i < 10; ++i)
          for (int j = i + 1; j < 10; ++j)
            if (p.params[i * 10 + j] > 0) edges.push(Value::Numbers{double(i), double(j), p.params[i * 10 + j]});
        for (int j = 0; j < 10; ++j) {
          const double w = o.mean(p, first, 1 + j);
          if (w > 0) edges.push(Value::Numbers{double(j), 10.0, w});
        }
        std::vector<double> recall;
        for (std::uint32_t cue = 0; cue < 11; ++cue) recall.push_back(o.mean(p, first + cue, 0));
        x.set("edges", edges).set("recallFrom", Value::Numbers(recall)).set("degree", o.mean(p, first, 11));
        return x;
      };
      return Value::object().set("schemaPlain", network(0)).set("schemaPrimed", network(11));
    };
    return m;
  }();
  return &model;
}

}  // namespace syn::models
