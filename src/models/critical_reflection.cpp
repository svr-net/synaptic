// Critical reflection (chapter 14) in the space: the assumptions in "good" practice, and for whom
// they do not hold.
//
// Agents are learners of five groups: confident learners at home with the norms of an American
// classroom; shy learners; learners new to those norms (for whom the teacher is the authority);
// learners still learning English; novices returning to learning. Each has a comfort point
// (affective channel), a sense of being entitled to question the teacher (social channel) and a
// readiness to direct their own learning (cognitive channel). A practice is a dilation of these
// points, with an effect that depends on the group (Brookfield; Shandomo): a circle of chairs
// invites the confident and exposes the shy and those new to the norms, less when the instructor
// explains why; calling on learners makes everyone speak and costs comfort, volunteers only lets
// the comfortable speak, a "pass" keeps both; an invitation to challenge the teacher is not
// believed until the teacher shows it (listening, taking ideas seriously, norms set together);
// maximum choice and minimum direction suits the self-directed and loses the novices, scaffolded
// choice serves both; search examples chosen without reflection carry the biases of the catalogue
// (Drabinski) and cost belonging to the learners they leave out. Speaking is a cap test of
// comfort; learning is a mixture towards the learner by the chance of comfort, by speaking and by
// how well the direction fits them. Jobs: the defaults, each alternative on its own, and all of
// them together.
#include <string>

#include "syn/models/models.hpp"

// The agent program, in WGSL.
constexpr const char* critical_agent = R"wgsl(
  let seating = u32(J(0u));
  let calling = u32(J(1u));
  let openness = u32(J(2u));
  let choice = u32(J(3u));
  let examples = J(4u) > f32(0.5);
  // The learner's group, by the shares of the five.
  let u = U(0u);
  var g = 4u;
  var found = false;
  var acc = f32(0);
  for (var k = 0u; k < 5u; k = k + 1u) {
    acc = acc + P(k);
    if (!found && u < acc) {
      g = k;
      found = true;
    }
  }
  // Group profiles: comfort, entitlement to question, self-direction (logits), and sensitivities.
  let t = 10u + g * 8u;
  var comfort = sampleLevel(U(1u), P(t), P(5u), kB);
  var entitled = sampleLevel(U(2u), P(t + 1u), P(5u), kC);
  let direction = sampleLevel(U(3u), P(t + 2u), P(5u), kA);
  let exposed = P(t + 3u);       // how much being on view costs
  let leftOut = P(t + 4u);       // how much unexamined examples cost belonging
  // Seating: 0 rows, 1 a circle, 2 a circle with the reason explained.
  if (seating > 0u) {
    let cost = exposed * select(f32(1), P(60u), seating == 2u);
    comfort = shift(comfort, P(61u) - cost);
  }
  // Examples chosen without reflection on whom they leave out.
  if (!examples) { comfort = shift(comfort, -leftOut); }
  // Calling: 0 called on, 1 volunteers only, 2 called on with a "pass".
  if (calling == 0u) { comfort = shift(comfort, -P(62u) * exposed); }
  if (calling == 2u) { comfort = shift(comfort, -P(63u) * exposed); }
  let willing = inside(comfort, chanceCap(U(4u)));
  var spoke = willing;
  if (calling == 0u) { spoke = true; }
  if (calling == 2u) { spoke = willing || inside(comfort, chanceCap(P(64u) * U(5u))); }
  // Power: 0 the teacher as expert, 1 an invitation to challenge, 2 the invitation shown in practice.
  if (openness == 1u) { entitled = shift(entitled, P(65u)); }
  if (openness == 2u) { entitled = shift(entitled, P(66u)); }
  let questioned = spoke && inside(entitled, chanceCap(U(6u)));
  // Direction: 0 directed, 1 scaffolded choice, 2 maximum choice, minimum direction.
  var fit = P(67u);
  if (choice == 1u) { fit = P(68u) + P(69u) * chance(direction); }
  if (choice == 2u) { fit = P(70u) * chance(direction); }
  var known = atChance(f32(0), kA);
  known = raise(known, P(71u) * fit * (P(72u) + chance(comfort)) * select(f32(1), P(73u), spoke) * select(f32(1), P(74u), questioned));
  let s = g * 5u;
  stat(s, f32(1));
  stat(s + 1u, select(f32(0), f32(1), spoke));
  stat(s + 2u, chance(comfort));
  stat(s + 3u, select(f32(0), f32(1), questioned));
  stat(s + 4u, chance(known));
)wgsl";

namespace syn::models {

namespace {
const char* kGroups[5] = {"confident, at home with the norms", "shy", "new to American classroom norms", "learning English", "novices returning to learning"};
// Comfort, entitlement to question, self-direction (logits); cost of being on view; cost of unexamined examples.
const double kProfile[5][5] = {{1.2, 0.3, 0.8, 0.0, 0.0}, {-0.2, -0.6, 0.3, 1.2, 0.1}, {0.0, -1.6, 0.0, 1.0, 0.6},
                               {-0.3, -0.9, 0.0, 0.8, 0.6}, {0.0, -0.4, -1.4, 0.4, 0.2}};
}  // namespace

const kernel::Model* criticalReflectionModel() {
  static const kernel::Model model = [] {
    kernel::Model m;
    m.name = "criticalReflection";
    m.title = "Critical reflection: seating, calling on learners, power, self-direction and examples, group by group";
    m.maxStats = 25;
    m.agentSource = critical_agent;
    m.compile = [](const Value& spec) {
      const Value& c = spec.sub("criticalReflection");
      std::vector<double> params = {c.num("confident", 0.45), c.num("shy", 0.2), c.num("newToNorms", 0.13), c.num("learningEnglish", 0.12),
                                    c.num("novices", 0.1), c.num("sd", 0.6), 0, 0, 0, 0};
      for (const auto& g : kProfile) {
        params.insert(params.end(), g, g + 5);
        params.insert(params.end(), {0, 0, 0});
      }
      params.resize(60, 0.0);
      params.insert(params.end(), {c.num("explained", 0.45), c.num("circle", 0.5), c.num("calledOn", 0.8), c.num("withPass", 0.15),
                                   c.num("passComfort", 0.6), c.num("invitation", 0.25), c.num("openness", 1.4), c.num("directed", 0.65),
                                   c.num("scaffolded", 0.6), c.num("scaffoldedChoice", 0.4), c.num("maximumChoice", 1.25),
                                   c.num("rate", 0.6), c.num("comfortFloor", 0.5), c.num("speaking", 1.3), c.num("questioning", 1.2)});
      const std::vector<std::vector<double>> rows = {{0, 0, 0, 0, 0}, {1, 0, 0, 0, 0}, {2, 0, 0, 0, 0}, {0, 1, 0, 0, 0}, {0, 2, 0, 0, 0}, {0, 0, 1, 0, 0},
                                                     {0, 0, 2, 0, 0}, {0, 0, 0, 2, 0}, {0, 0, 0, 1, 0}, {0, 0, 0, 0, 1}, {2, 2, 2, 1, 1}};
      return std::vector<Plan>{makePlan("criticalReflection", learnersOf(spec, 2000), 25, params, rows, seedOf(spec))};
    };
    m.summarise = [](const Value&, const std::vector<Plan>& plans, const std::vector<Output>& out) {
      const Plan& p = plans[0];
      const Output& o = out[0];
      const char* names[] = {"defaults: rows, calling on learners, teacher as expert, directed, unexamined examples",
                             "a circle", "a circle, with the reason explained", "volunteers only", "calling on learners with a \"pass\"",
                             "an invitation to challenge the teacher", "the invitation shown in practice (listening, shared norms)",
                             "maximum choice, minimum direction", "scaffolded choice", "examples chosen critically", "critically reflective: all of the above"};
      Value designs = Value::array();
      for (std::uint32_t j = 0; j < p.jobs; ++j) {
        Value groups = Value::array();
        double total = 0, spoke = 0, comfort = 0, questioned = 0, learned = 0, lo = 1, hi = 0;
        for (std::uint32_t g = 0; g < 5; ++g) {
          const double share = o.mean(p, j, g * 5);
          auto per = [&](std::uint32_t k) { return share > 0 ? o.mean(p, j, g * 5 + k) / share : 0.0; };
          groups.push(Value::object().set("group", kGroups[g]).set("share", share).set("spoke", per(1)).set("comfort", per(2)).set("questioned", per(3)).set("learned", per(4)));
          total += share;
          spoke += o.mean(p, j, g * 5 + 1);
          comfort += o.mean(p, j, g * 5 + 2);
          questioned += o.mean(p, j, g * 5 + 3);
          learned += o.mean(p, j, g * 5 + 4);
          if (share > 0) {
            lo = std::min(lo, per(4));
            hi = std::max(hi, per(4));
          }
        }
        designs.push(Value::object()
                         .set("name", names[j])
                         .set("groups", groups)
                         .set("spoke", spoke / total)
                         .set("comfort", comfort / total)
                         .set("questioned", questioned / total)
                         .set("learned", learned / total)
                         .set("lowestGroup", lo)
                         .set("gap", hi - lo));
      }
      return Value::object()
          .set("designs", designs)
          .set("groups", std::vector<std::string>(kGroups, kGroups + 5))
          .set("assumptions", std::vector<std::string>{"Lecturing is passive and boring; discussions and activities are better.",
                                                       "The teacher as expert discourages questions; position yourself as a collaborator.",
                                                       "Information literacy is power: everyone needs it, and it narrows gaps.",
                                                       "Learners want to direct themselves: maximum choice, minimum direction."});
    };
    return m;
  }();
  return &model;
}

}  // namespace syn::models
