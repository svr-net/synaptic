// Developmental stages (Piaget, Perry) in the space.
//
// Piaget: a child reaches stage k when its age passes the stage's onset, which varies between
// children; the chance of having reached it is the chance of the point at logit
// (age − onset)/spread, and P(stage = k) = reached(k) − reached(k + 1). An activity fits a child
// whose stage provides all its demands: the chance of having reached the activity's first stage.
// Perry: a student's position (dualism, multiplicity, relativism, commitment) moves on each term
// when the point of the transition chance holds a uniform draw; instruction that challenges
// students to evaluate sources dilates the transitions out of multiplicity and relativism.
// Judging sources: each source's evidence quality, authority cue and personal resonance are
// levels; a student's judgement dilates the equator point by their weighted logits, and accuracy
// is the share of source pairs ordered like their quality.
#include <cmath>
#include <string>

#include "syn/models/models.hpp"

// The agent program, in WGSL.
constexpr const char* development_agent = R"wgsl(
  let kind = J(0u);
  if (kind < f32(0.5)) {
    let age = J(1u);
    let spread = P(0u) * kLogisticUnit;
    var reached : array<f32, 5>;
    reached[0] = f32(1);
    for (var k = 1u; k < 4u; k = k + 1u) { reached[k] = chance(atLogit((age - P(k)) / spread, kA)); }
    reached[4] = f32(0);
    for (var k = 0u; k < 4u; k = k + 1u) { stat(k, max(f32(0), reached[k] - reached[k + 1u])); }
    for (var a = 0u; a < 6u; a = a + 1u) { stat(4u + a, reached[u32(P(4u + a))]); }
  } else if (kind < f32(1.5)) {
    let terms = u32(P(10u));
    let challenge = clamp(J(1u), f32(0), f32(1));
    var position = 0u;
    stat(0u, f32(1));
    for (var t = 0u; t < terms; t = t + 1u) {
      if (position < 3u) {
        let boost = select(f32(1), f32(1) + P(14u) * challenge, position >= 1u);
        let advance = atChance(clamp(P(11u + position) * boost, f32(0), f32(1)), kA);
        if (inside(advance, chanceCap(U(t)))) { position = position + 1u; }
      }
      stat((t + 1u) * 4u + position, f32(1));
    }
  } else {
    let position = u32(J(1u));
    let rho = clamp(P(15u), f32(-1), f32(1));
    let wAuthority = P(16u + 3u * position);
    let wPersonal = P(17u + 3u * position);
    let wEvidence = P(18u + 3u * position);
    var quality : array<f32, 8>;
    var judged : array<f32, 8>;
    for (var s = 0u; s < 8u; s = s + 1u) {
      let q = sampleLevel(U(4u * s), f32(0), f32(1), kA);
      let z = kLogisticUnit * logisticOf(U(4u * s + 1u));
      let authority = rho * logit(q) + sqrt(f32(1) - rho * rho) * z;
      let personal = kLogisticUnit * logisticOf(U(4u * s + 2u));
      let noise = f32(0.2) * kLogisticUnit * logisticOf(U(4u * s + 3u));
      let judgement = shift(shift(shift(shift(equator(kC), wAuthority * authority), wPersonal * personal), wEvidence * logit(q)), noise);
      quality[s] = chance(q);
      judged[s] = chance(judgement);
    }
    var agree = f32(0);
    var pairs = f32(0);
    for (var i = 0u; i < 8u; i = i + 1u) {
      for (var j = i + 1u; j < 8u; j = j + 1u) {
        if ((quality[i] - quality[j]) * (judged[i] - judged[j]) > f32(0)) { agree = agree + f32(1); }
        pairs = pairs + f32(1);
      }
    }
    stat(0u, agree / pairs);
  }
)wgsl";

namespace syn::models {

namespace {
struct Activity {
  const char* name;
  std::vector<const char*> demands;
  int firstStage;  // the first stage that provides every demand
};
const std::vector<Activity>& lesson() {
  static const std::vector<Activity> a = {
      {"look at pictures of animal habitats", {"language"}, 1},
      {"nature walk: describe what you see", {"language", "trial-and-error problem solving"}, 1},
      {"reflect on your own experience of weather", {"language", "imaginative play"}, 1},
      {"imagine how changes affect other people and organisms", {"other people's perspectives", "logical thinking"}, 2},
      {"anticipate consequences of climate change", {"abstract thinking", "thinking through consequences"}, 3},
      {"propose steps to improve the environment", {"deductive reasoning", "thinking through consequences", "other people's perspectives"}, 3},
  };
  return a;
}
}  // namespace

const kernel::Model* developmentModel() {
  static const kernel::Model model = [] {
    kernel::Model m;
    m.name = "development";
    m.title = "Piaget's stages by age and activity fit; Perry's positions and judging sources";
    m.maxStats = 68;
    m.agentSource = development_agent;
    m.compile = [](const Value& spec) {
      const Value& c = spec.sub("development");
      const int terms = std::min(15, c.integer("terms", 8));
      std::vector<double> params = {c.num("onsetSd", 0.9), 2.0, 7.0, 12.0};
      for (const auto& a : lesson()) params.push_back(a.firstStage);
      params.insert(params.end(), {double(terms), 0.12, 0.08, 0.05, 1.5, c.num("authorityValidity", 0.3)});
      // Judgement weights (authority, personal resonance, evidence) by position.
      params.insert(params.end(), {0.85, 0.10, 0.05, 0.10, 0.85, 0.05, 0.20, 0.10, 0.70, 0.15, 0.05, 0.80});
      std::vector<std::vector<double>> ages, perry, sources;
      for (double age : grid(0, 18, 181)) ages.push_back({0, age});
      perry = {{1, 0.0}, {1, c.num("challenge", 1.0)}};
      for (int k = 0; k < 4; ++k) sources.push_back({2, double(k)});
      return std::vector<Plan>{makePlan("development", 1, 10, params, ages, seedOf(spec)),
                               makePlan("development", static_cast<std::uint32_t>(c.integer("students", 4000)), 4 * (terms + 1), params, perry, seedOf(spec)),
                               makePlan("development", static_cast<std::uint32_t>(c.integer("trials", 400)), 1, params, sources, seedOf(spec))};
    };
    m.summarise = [](const Value&, const std::vector<Plan>& plans, const std::vector<Output>& out) {
      Value r = Value::object();
      const Plan& p = plans[0];
      r.set("ages", Value::Numbers(grid(0, 18, 181)));
      Value dist = Value::array();
      for (std::uint32_t k = 0; k < 4; ++k) dist.push(Value::Numbers(column(p, out[0], k)));
      r.set("stageDistribution", dist);
      const char* names[] = {"sensorimotor", "preoperational", "concrete operational", "formal operational"};
      const char* ages[] = {"birth to 18-24 months", "18-24 months to 7 years", "7 to 12 years", "12 years and up"};
      Value stages = Value::array();
      for (int k = 0; k < 4; ++k) stages.push(Value::object().set("name", names[k]).set("ages", ages[k]));
      r.set("stages", stages);
      Value acts = Value::array();
      for (std::size_t a = 0; a < lesson().size(); ++a) {
        std::vector<std::string> demands(lesson()[a].demands.begin(), lesson()[a].demands.end());
        acts.push(Value::object().set("name", lesson()[a].name).set("fit", Value::Numbers(column(p, out[0], 4 + static_cast<std::uint32_t>(a)))).set("demands", demands));
      }
      r.set("activities", acts);
      const Plan& q = plans[1];
      const int terms = static_cast<int>(q.params[10]);
      for (std::uint32_t j = 0; j < 2; ++j) {
        Value byPos = Value::array();
        for (int k = 0; k < 4; ++k) {
          std::vector<double> v;
          for (int t = 0; t <= terms; ++t) v.push_back(out[1].mean(q, j, t * 4 + k));
          byPos.push(Value::Numbers(v));
        }
        r.set(j == 0 ? "perryPlain" : "perryChallenged", byPos);
      }
      r.set("perryNames", std::vector<std::string>{"dualism", "multiplicity", "relativism", "commitment"});
      r.set("sourceAccuracy", Value::Numbers(column(plans[2], out[2], 0)));
      return r;
    };
    return m;
  }();
  return &model;
}

}  // namespace syn::models
