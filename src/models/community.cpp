// Community data and its limits (chapter 7) in the space.
//
// A learner has each demographic attribute (nontraditional, first-generation, English learner,
// older adult, disability) when the point of the attribute's prevalence holds a uniform draw;
// their prior knowledge is a level dilated by the effects of their attributes and an individual
// deviation, and a pre-assessment reads it with noise. The kernel accumulates the sums from which
// the variance of prior knowledge explained by each label, by every intersection of labels, by
// the pre-assessment and by both together follows, and the cell sizes behind aggregate-only
// reporting (cells smaller than k are suppressed).
#include <string>

#include "syn/models/models.hpp"

// The agent program, in WGSL.
constexpr const char* community_agent = R"wgsl(
  let k = 5u;
  var mask = 0u;
  var effect = f32(0);
  for (var a = 0u; a < k; a = a + 1u) {
    if (inside(atChance(P(2u + 2u * a), kC), chanceCap(U(a)))) {
      mask = mask | (1u << a);
      effect = effect + P(3u + 2u * a);
    }
  }
  let theta = logit(sampleLevel(U(10u), effect, P(0u), kA));
  let measured = theta + P(1u) * kLogisticUnit * logisticOf(U(11u));
  stat(0u, theta);
  stat(1u, theta * theta);
  stat(2u, measured);
  stat(3u, measured * measured);
  stat(4u, theta * measured);
  for (var a = 0u; a < k; a = a + 1u) {
    if ((mask & (1u << a)) != 0u) {
      stat(8u + 2u * a, theta);
      stat(9u + 2u * a, f32(1));
    }
  }
  stat(32u + 3u * mask, f32(1));
  stat(33u + 3u * mask, theta);
  stat(34u + 3u * mask, measured);
)wgsl";

namespace syn::models {

namespace {
struct Attribute {
  const char* name;
  double prevalence, effect;
};
const Attribute kAttributes[5] = {{"nontraditional", 0.7, -0.15}, {"first-generation", 0.3, -0.35}, {"English learner", 0.15, -0.4},
                                  {"older adult", 0.2, -0.25}, {"disability", 0.12, 0.0}};
}  // namespace

const kernel::Model* communityModel() {
  static const kernel::Model model = [] {
    kernel::Model m;
    m.name = "community";
    m.title = "Community data: labels, intersections and pre-assessment as predictors; minimum cell sizes";
    m.maxStats = 128;
    m.agentSource = community_agent;
    m.compile = [](const Value& spec) {
      const Value& c = spec.sub("community");
      std::vector<double> params = {c.num("individualSd", 0.9), c.num("measurementNoise", 0.6)};
      for (const auto& a : kAttributes) params.insert(params.end(), {a.prevalence, a.effect});
      return std::vector<Plan>{makePlan("community", static_cast<std::uint32_t>(c.integer("learners", 3000)), 128, params, {{0.0}}, seedOf(spec))};
    };
    m.summarise = [](const Value& spec, const std::vector<Plan>& plans, const std::vector<Output>& out) {
      const Value& c = spec.sub("community");
      const Plan& p = plans[0];
      const Output& o = out[0];
      const double n = p.agents;
      auto S = [&](std::uint32_t i) { return o.mean(p, 0, i) * n; };
      const double st = S(0), stt = S(1), sm = S(2), smm = S(3), stm = S(4);
      const double mean = st / n, sst = stt - n * mean * mean;
      Value r = Value::object();
      std::vector<std::string> names;
      std::vector<double> share, groupMean, single;
      double best = 0;
      for (std::uint32_t a = 0; a < 5; ++a) {
        const double s1 = S(8 + 2 * a), n1 = std::round(S(9 + 2 * a)), s0 = st - s1, n0 = n - n1;
        const double m1 = n1 > 0 ? s1 / n1 : 0, m0 = n0 > 0 ? s0 / n0 : 0;
        const double r2 = sst > 0 ? (n1 * (m1 - mean) * (m1 - mean) + n0 * (m0 - mean) * (m0 - mean)) / sst : 0;
        names.push_back(kAttributes[a].name);
        share.push_back(n1 / n);
        groupMean.push_back(m1);
        single.push_back(r2);
        best = std::max(best, r2);
      }
      r.set("attributes", names).set("share", Value::Numbers(share)).set("mean", Value::Numbers(groupMean)).set("singleR2", Value::Numbers(single));
      r.set("bestSingleR2", best);
      const double s2 = std::pow(c.num("individualSd", 0.9), 2), e2 = std::pow(c.num("measurementNoise", 0.6), 2);
      const double w1 = (1 / s2) / (1 / s2 + 1 / e2), w2 = 1 - w1, rel = s2 / (s2 + e2);
      double between = 0, sseCombined = stt - 2 * w2 * stm + w2 * w2 * smm;
      int cells = 0;
      std::vector<double> counts;
      for (std::uint32_t mask = 0; mask < 32; ++mask) {
        const double cn = std::round(S(32 + 3 * mask)), ct = S(33 + 3 * mask), cm = S(34 + 3 * mask);
        if (cn < 0.5) continue;
        ++cells;
        counts.push_back(cn);
        const double cellMean = ct / cn;
        between += cn * (cellMean - mean) * (cellMean - mean);
        sseCombined += -2 * w1 * cellMean * ct + w1 * w1 * cn * cellMean * cellMean + 2 * w1 * w2 * cellMean * cm;
      }
      r.set("intersectionR2", sst > 0 ? between / sst : 0.0);
      r.set("preAssessmentR2", sst > 0 ? 1 - (stt - 2 * rel * stm + rel * rel * smm) / sst : 0.0);
      r.set("combinedR2", sst > 0 ? 1 - sseCombined / sst : 0.0);
      r.set("cells", cells);
      std::vector<double> ks = {1, 5, 10, 20, 50}, cellsShare, learnersShare;
      for (double k : ks) {
        double small = 0, covered = 0;
        for (double cn : counts)
          if (cn < k) {
            small += 1;
            covered += cn;
          }
        cellsShare.push_back(cells ? small / cells : 0);
        learnersShare.push_back(covered / n);
      }
      r.set("minCellSizes", Value::Numbers(ks)).set("suppressedCells", Value::Numbers(cellsShare)).set("suppressedLearners", Value::Numbers(learnersShare));
      (void)sm;
      return r;
    };
    return m;
  }();
  return &model;
}

}  // namespace syn::models
