// Scaffolding (chapter 10) in the space: Ellis and Larkin's four steps, and projects in stages.
//
// A learner's skill is a point of the cognitive channel; a task is the point of their skill
// against its difficulty. Support is a dilation of the task: large when the instructor leads
// the whole class ("we do it"), smaller when partners practise together and give each other
// feedback ("you do it together"), none alone ("you do it alone"). The three zones of the ZPD are
// caps around the learner, as on the social constructivism page: in zone 1 (they can already do
// it alone) practice teaches little; in zone 2 a supported success teaches most; in zone 3
// (beyond them even with the support they have) a failure adds to their frustration, a point of
// the affective channel, and a learner whose frustration holds a uniform draw stops trying. A
// demonstration teaches by watching. Jobs: the four steps and the shortcuts around them, each
// with the same minutes; and a research project done in one go, or in stages (topic, annotated
// bibliography, outline, paper), with or without feedback at each stage. Agents: learners.
#include <string>

#include "syn/models/models.hpp"

// The agent program, in WGSL.
constexpr const char* scaffolding_agent = R"wgsl(
  let project = J(0u) > f32(0.5);
  var skill = sampleLevel(U(0u), P(0u), P(1u), kA);
  let start = logit(skill);
  var frustration = atChance(P(2u), kB);
  let difficulty = P(3u);
  if (project) {
    // A project in stages: each stage's work is the chance of the task; feedback on a stage
    // corrects it and teaches; without stages, the errors of each part carry into the next.
    let stages = u32(J(1u));
    let feedback = J(2u) > f32(0.5);
    // Stages spread the work over the term instead of the night before the deadline.
    let spread = select(f32(0), P(14u), stages > 1u);
    var quality = atChance(f32(1), kA);
    for (var s = 0u; s < 4u; s = s + 1u) {
      let task = atLogit(logit(skill) - P(16u) - P(10u) * f32(s) + spread, kA);
      let done = chance(task);
      let checked = stages > 1u && feedback;
      // A stage that gets feedback is revised: most of its shortfall is repaired.
      let kept = select(done, done + P(11u) * (f32(1) - done), checked);
      quality = lower(quality, f32(1) - kept);
      if (checked) { skill = shift(skill, P(12u) * (f32(1) - done)); }
      stat(4u + s, kept);
    }
    stat(0u, chance(quality));
    stat(1u, logit(skill) - start);
    return;
  }
  // Minutes of each step: demonstrate, we do it, you do it together, you do it alone.
  var minutes : array<f32, 4>;
  for (var k = 0u; k < 4u; k = k + 1u) { minutes[k] = J(1u + k); }
  var support : array<f32, 4>;
  support[0] = f32(0);
  support[1] = P(4u);
  support[2] = P(5u);
  support[3] = f32(0);
  var stopped = false;
  var idx = 10u;
  for (var phase = 0u; phase < 4u; phase = phase + 1u) {
    let attempts = u32(minutes[phase] / P(6u));
    for (var a = 0u; a < attempts; a = a + 1u) {
      idx = idx + 2u;
      if (phase == 0u) {
        // Watching a demonstration: learning by observation, more for whoever can follow it.
        skill = shift(skill, P(7u) * chance(atLogit(logit(skill) - difficulty + P(4u), kA)) * f32(0.5));
        continue;
      }
      if (stopped) { continue; }
      let alone = atLogit(logit(skill) - difficulty, kA);
      let helped = shift(alone, support[phase]);
      let zone1 = inside(alone, chanceCap(P(8u)));
      let zone3 = !inside(helped, chanceCap(P(9u)));
      stat(8u + select(select(1u, 2u, zone3), 0u, zone1), f32(1));
      let success = inside(helped, chanceCap(U(idx)));
      if (success) {
        // A success teaches what the learner could not yet do alone (little in zone 1); what the
        // support carried transfers to working alone only in part, less the more support there was
        // (scaffolds have to fade).
        let carried = exp(-P(15u) * support[phase]);
        skill = shift(skill, P(7u) * (f32(1) - chance(alone)) * carried);
        stat(11u, f32(1));
      } else {
        if (phase == 2u) { skill = shift(skill, P(7u) * f32(0.3) * (f32(1) - chance(alone))); }  // a partner's feedback
        if (zone3) { frustration = raise(frustration, P(13u)); }
      }
      if (inside(frustration, chanceCap(U(idx + 1u))) && zone3) { stopped = true; }
    }
  }
  // The transfer task, alone: a new search of the same kind.
  stat(0u, chance(atLogit(logit(skill) - difficulty - P(10u), kA)));
  stat(1u, logit(skill) - start);
  stat(2u, select(f32(0), f32(1), stopped));
  stat(3u, chance(frustration));
)wgsl";

namespace syn::models {

const kernel::Model* scaffoldingModel() {
  static const kernel::Model model = [] {
    kernel::Model m;
    m.name = "scaffolding";
    m.title = "Scaffolding: Ellis and Larkin's four steps against shortcuts; projects in stages with feedback";
    m.maxStats = 12;
    m.agentSource = scaffolding_agent;
    m.compile = [](const Value& spec) {
      const Value& c = spec.sub("scaffolding");
      const std::vector<double> params = {c.num("skillMean", -0.5), c.num("skillSd", 1.0), c.num("frustration", 0.1), c.num("difficulty", 1.0),
                                          c.num("guidedSupport", 2.5), c.num("pairSupport", 1.2), c.num("minutesPerAttempt", 1.5),
                                          c.num("learningRate", 0.35), c.num("zoneInner", 0.8), c.num("zoneOuter", 0.25), c.num("transfer", 0.3),
                                          c.num("revision", 0.7), c.num("feedbackLearning", 0.4), c.num("frustrationStep", 0.15),
                                          c.num("stagingTime", 0.3), c.num("dependence", 0.45), c.num("projectDifficulty", -1.2)};
      const double total = c.num("minutes", 24.0);
      const std::vector<std::vector<double>> steps = {{0, 0, 0, 0, total},
                                                     {0, 4, 0, 0, total - 4},
                                                     {0, 4, total - 4, 0, 0},
                                                     {0, 4, (total - 4) / 3, (total - 4) / 3, (total - 4) / 3},
                                                     {0, 4, (total - 4) / 2, (total - 4) / 2, 0},
                                                     {1, 1, 0, 0, 0},
                                                     {1, 4, 0, 0, 0},
                                                     {1, 4, 1, 0, 0}};
      return std::vector<Plan>{makePlan("scaffolding", learnersOf(spec, 500), 12, params, steps, seedOf(spec))};
    };
    m.summarise = [](const Value&, const std::vector<Plan>& plans, const std::vector<Output>& out) {
      const Plan& p = plans[0];
      const Output& o = out[0];
      const char* names[] = {"on your own from the start",
                             "demonstration, then on your own",
                             "demonstration, then the instructor leads it all",
                             "gradual release: I do, we do, you do together, you do alone",
                             "gradual release without practice alone"};
      Value sequences = Value::array(), projects = Value::array();
      for (std::uint32_t j = 0; j < 5; ++j) {
        const double zones = o.mean(p, j, 8) + o.mean(p, j, 9) + o.mean(p, j, 10);
        sequences.push(Value::object()
                           .set("name", names[j])
                           .set("minutes", Value::Numbers(std::vector<double>(p.jobRows.begin() + j * p.jobStride + 1, p.jobRows.begin() + j * p.jobStride + 5)))
                           .set("transfer", o.mean(p, j, 0))
                           .set("gain", o.mean(p, j, 1))
                           .set("gaveUp", o.mean(p, j, 2))
                           .set("frustration", o.mean(p, j, 3))
                           .set("zones", Value::Numbers(zones > 0 ? std::vector<double>{o.mean(p, j, 8) / zones, o.mean(p, j, 9) / zones, o.mean(p, j, 10) / zones}
                                                                  : std::vector<double>{0, 0, 0}))
                           .set("successes", o.mean(p, j, 11)));
      }
      const char* projectNames[] = {"one deadline, no stages", "four stages, no feedback", "four stages with feedback"};
      for (std::uint32_t j = 5; j < 8; ++j)
        projects.push(Value::object()
                          .set("name", projectNames[j - 5])
                          .set("quality", o.mean(p, j, 0))
                          .set("gain", o.mean(p, j, 1))
                          .set("stages", Value::Numbers(o.range(p, j, 4, 4))));
      return Value::object()
          .set("sequences", sequences)
          .set("projects", projects)
          .set("steps", std::vector<std::string>{"I do it (demonstrate)", "we do it (whole class)", "you do it together (pairs)", "you do it alone"})
          .set("stageNames", std::vector<std::string>{"topic", "annotated bibliography", "outline", "paper"});
    };
    return m;
  }();
  return &model;
}

}  // namespace syn::models
