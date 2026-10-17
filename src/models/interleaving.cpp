// Blocked and interleaved practice (cognitivism) in the space.
//
// The skill of executing each of K related procedures (AND, OR, NOT searches) is a point of the
// cognitive channel raised towards the learner by every practice problem: raise(E, η). Telling the
// procedures apart is a point D whose chance is 1 − e^{−d}: each problem that forces a choice
// adds δ to d, which is the mixture raise(D, 1 − e^{−δ}). Accuracy on a problem is the chance of
// choosing the right procedure, 1/K + (1 − 1/K)·chance(D), times the chance of executing it.
// Blocked practice only chooses at the first problem of each block. Jobs: blocked, interleaved
// (the same learners: common random numbers); agents: learners with log-logistic rates.
#include "syn/models/models.hpp"

// The agent program, in WGSL.
constexpr const char* interleaving_agent = R"wgsl(
  let categories = u32(P(0u));
  let per = u32(P(1u));
  let interleaved = J(0u) > f32(0.5);
  let eta = min(f32(1), P(3u) * exp(P(5u) * kLogisticUnit * logisticOf(U(0u))));
  let delta = P(4u) * exp(P(5u) * kLogisticUnit * logisticOf(U(1u)));
  let guess = f32(1) / f32(categories);
  var skill : array<V5, 8>;
  for (var k = 0u; k < categories; k = k + 1u) { skill[k] = atChance(P(2u), kA); }
  var discrimination = atChance(f32(0), kA);
  let n = categories * per;
  for (var t = 0u; t < n; t = t + 1u) {
    let kind = select(t / per, t % categories, interleaved);
    let choose = interleaved || t % per == 0u;
    let pick = select(f32(1), guess + (f32(1) - guess) * chance(discrimination), choose);
    stat(t, pick * chance(skill[kind]));
    if (choose) { discrimination = raise(discrimination, f32(1) - exp(-delta)); }
    skill[kind] = raise(skill[kind], eta);
  }
  var execution = f32(0);
  for (var k = 0u; k < categories; k = k + 1u) { execution = execution + chance(skill[k]) / f32(categories); }
  let pickTest = guess + (f32(1) - guess) * chance(discrimination);
  stat(64u, pickTest);
  stat(65u, execution);
  stat(66u, pickTest * execution);
)wgsl";

namespace syn::models {

const kernel::Model* interleavingModel() {
  static const kernel::Model model = [] {
    kernel::Model m;
    m.name = "interleaving";
    m.title = "Blocked and interleaved practice: execution and discrimination as mixtures";
    m.maxStats = 67;
    m.agentSource = interleaving_agent;
    m.compile = [](const Value& spec) {
      const Value& c = spec.sub("interleaving");
      const int k = std::max(2, std::min(8, c.integer("categories", 3)));
      const int per = std::max(1, std::min(64 / k, c.integer("problemsPerCategory", 12)));
      const std::vector<double> params = {double(k), double(per), c.num("initialExecution", 0.3), c.num("executionRate", 0.12),
                                          c.num("discriminationRate", 0.12), c.num("heterogeneity", 0.3)};
      return std::vector<Plan>{makePlan("interleaving", static_cast<std::uint32_t>(c.integer("learners", 400)), 67, params, {{0.0}, {1.0}}, seedOf(spec))};
    };
    m.summarise = [](const Value&, const std::vector<Plan>& plans, const std::vector<Output>& out) {
      const Plan& p = plans[0];
      const Output& o = out[0];
      const int n = static_cast<int>(p.params[0] * p.params[1]);
      Value r = Value::object();
      const char* names[] = {"blocked", "interleaved"};
      for (std::uint32_t j = 0; j < 2; ++j) {
        const auto practice = o.range(p, j, 0, n);
        double meanPractice = 0;
        for (double a : practice) meanPractice += a / n;
        r.set(names[j], Value::object()
                            .set("practice", Value::Numbers(practice))
                            .set("practiceMean", meanPractice)
                            .set("choice", o.mean(p, j, 64))
                            .set("execution", o.mean(p, j, 65))
                            .set("test", o.mean(p, j, 66)));
      }
      return r;
    };
    return m;
  }();
  return &model;
}

}  // namespace syn::models
