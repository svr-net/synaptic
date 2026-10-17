// Lesson plans and time management (chapter 10) in the space.
//
// An agent is one delivery of a lesson plan — by default the chapter's Example 10.2, "Fighting
// fake news" (60 minutes in a public library: welcome, lecture, activity, debrief, wrap-up,
// reflection). How long a segment really takes is its planned time scaled by the exponential of
// a random level (the logistic of the space) around a bias: new instructors underestimate, most
// of all for activities and discussion; without a written plan the drift is larger and a
// segment can be forgotten. A plan with time stamps lets the instructor check the clock at each
// segment: behind, with a cut list, they cut the optional part of the long segments; without one
// they rush the rest (it teaches less). The session stops at the hour, so whatever is left at the
// end — the wrap-up that returns to the outcomes, the reflection — is lost. Ahead, a backup
// activity uses the spare minutes (two to five). Each segment delivered moves the learners' knowledge towards
// them by its value, as a mixture. Jobs: new and experienced instructors, without a plan, with a
// timed plan, with a cut list, with a backup activity, with both.
#include <string>

#include "syn/models/models.hpp"

// The agent program, in WGSL.
constexpr const char* plan_agent = R"wgsl(
  let experienced = J(0u) > f32(0.5);
  let planned = J(1u) > f32(0.5);
  let cutList = J(2u) > f32(0.5);
  let backup = J(3u) > f32(0.5);
  let segments = u32(P(0u));
  let limit = P(1u);
  let bias = select(P(2u), P(3u), experienced) + select(P(4u), f32(0), planned);
  let sd = P(5u) + select(P(6u), f32(0), planned);
  var known = atChance(f32(0), kA);
  var clock = f32(0);
  var plannedClock = f32(0);
  var rushed = f32(0);
  var cut = f32(0);
  var lostSegments = f32(0);
  var complete = true;
  var wrapped = true;
  for (var k = 0u; k < segments; k = k + 1u) {
    let minutes = P(10u + 4u * k);
    let value = P(11u + 4u * k);
    let activity = P(12u + 4u * k) > f32(0.5);
    let optional = P(13u + 4u * k);
    // Without a written plan, a segment is sometimes forgotten.
    if (!planned && inside(atChance(P(7u), kC), chanceCap(U(100u + k)))) {
      lostSegments = lostSegments + f32(1);
      plannedClock = plannedClock + minutes;
      complete = false;
      if (k + 2u >= segments) { wrapped = false; }
      continue;
    }
    // How long it really takes.
    let drift = logit(sampleLevel(U(k), bias + select(f32(0), P(8u), activity && !experienced), sd, kA));
    var takes = minutes * exp(drift);
    var factor = f32(1);
    if (planned) {
      // Watching the clock: how far behind the plan this segment will end.
      let behind = clock + takes - (plannedClock + minutes);
      if (behind > P(9u)) {
        if (cutList && optional > f32(0)) {
          let saved = min(behind, optional * takes);
          takes = takes - saved;
          cut = cut + saved;
          factor = f32(1) - P(30u) * saved / max(minutes, f32(1));
        } else {
          // Rushing: the segment is squeezed, and teaches less.
          let squeeze = min(P(34u), behind / max(takes, f32(1)));
          takes = takes * (f32(1) - squeeze);
          factor = f32(1) - P(31u) * squeeze;
          rushed = rushed + f32(1);
        }
      }
    }
    let left = limit - clock;
    let delivered = clamp(left / takes, f32(0), f32(1));
    known = raise(known, value * factor * delivered);
    stat(10u + k, delivered);
    if (delivered < f32(0.999)) {
      complete = false;
      if (k + 2u >= segments) { wrapped = false; }
    }
    clock = clock + min(takes, max(left, f32(0)));
    plannedClock = plannedClock + minutes;
  }
  // Spare minutes: a backup activity, or an early end.
  var spare = max(limit - clock, f32(0));
  if (backup && spare >= f32(2)) {
    let used = min(spare, P(32u));
    known = raise(known, P(33u) * used / P(32u));
    spare = spare - used;
  }
  // On time: the session ended with its wrap-up and reflection, in full.
  stat(0u, select(f32(0), f32(1), wrapped));
  stat(7u, select(f32(0), f32(1), complete));
  stat(1u, max(plannedClock - limit, f32(0)));
  stat(2u, chance(known));
  stat(3u, spare);
  stat(4u, cut);
  stat(5u, rushed);
  stat(6u, lostSegments);
)wgsl";

namespace syn::models {

namespace {
struct Segment {
  const char* name;
  double minutes, value;
  bool activity;
  double optional;  // share that can be cut
  const char* content;
};
// Example 10.2, "Fighting fake news" (60 minutes, public library, general adults).
const Segment kExample[6] = {
    {"Welcome", 5, 0.05, false, 0.0, "Introductions; the outcomes of the session"},
    {"Lecture", 10, 0.2, false, 0.3, "Opening questions on impact and trusted sources; history, current landscape, definitions"},
    {"Activity", 25, 0.35, true, 0.4, "Read two excerpts (5); pairs judge trustworthiness and what is missing (10); pairs build a criteria list on the flip chart (10)"},
    {"Debrief", 10, 0.2, true, 0.3, "Review the criteria; apply them to a new excerpt; which stories are worth sharing"},
    {"Wrap-up", 5, 0.12, false, 0.0, "Return to the outcomes and the takeaways"},
    {"Reflection", 5, 0.12, true, 0.0, "Minute paper; answer the questions"}};
}  // namespace

const kernel::Model* lessonPlanModel() {
  static const kernel::Model model = [] {
    kernel::Model m;
    m.name = "lessonPlan";
    m.title = "Lesson plans: Example 10.2 delivered with and without a timed plan, a cut list and a backup activity";
    m.maxStats = 16;
    m.agentSource = plan_agent;
    m.compile = [](const Value& spec) {
      const Value& c = spec.sub("lessonPlan");
      std::vector<double> params = {6, c.num("minutes", 60.0), c.num("newBias", 0.12), c.num("experiencedBias", 0.02), c.num("noPlanBias", 0.08),
                                    c.num("drift", 0.12), c.num("noPlanDrift", 0.1), c.num("forgotten", 0.06), c.num("activityBias", 0.15),
                                    c.num("checkMinutes", 1.5)};
      for (const auto& s : kExample) params.insert(params.end(), {s.minutes, s.value, s.activity ? 1.0 : 0.0, s.optional});
      params.resize(30, 0.0);
      params.insert(params.end(), {c.num("cutCost", 0.3), c.num("rushCost", 0.8), c.num("backupMinutes", 5.0), c.num("backupValue", 0.08),
                                     c.num("maxRush", 0.25)});
      std::vector<std::vector<double>> rows;
      for (double skilled : {0.0, 1.0})
        for (const auto& f : std::vector<std::vector<double>>{{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {1, 0, 1}, {1, 1, 1}}) rows.push_back({skilled, f[0], f[1], f[2]});
      return std::vector<Plan>{makePlan("lessonPlan", static_cast<std::uint32_t>(c.integer("deliveries", 2000)), 16, params, rows, seedOf(spec))};
    };
    m.summarise = [](const Value&, const std::vector<Plan>& plans, const std::vector<Output>& out) {
      const Plan& p = plans[0];
      const Output& o = out[0];
      const char* names[] = {"no written plan", "timed plan", "timed plan + cut list", "timed plan + backup activity", "timed plan + cut list + backup"};
      Value designs = Value::array();
      for (std::uint32_t j = 0; j < p.jobs; ++j)
        designs.push(Value::object()
                         .set("instructor", j < 5 ? "new" : "experienced")
                         .set("name", names[j % 5])
                         .set("onTime", o.mean(p, j, 0))
                         .set("overrunPlanned", o.mean(p, j, 1))
                         .set("learning", o.mean(p, j, 2))
                         .set("spareMinutes", o.mean(p, j, 3))
                         .set("cutMinutes", o.mean(p, j, 4))
                         .set("rushedSegments", o.mean(p, j, 5))
                         .set("forgottenSegments", o.mean(p, j, 6))
                         .set("allInFull", o.mean(p, j, 7))
                         .set("delivered", Value::Numbers(o.range(p, j, 10, 6))));
      Value plan = Value::array();
      double at = 0;
      for (const auto& s : kExample) {
        plan.push(Value::object().set("segment", s.name).set("start", at).set("minutes", s.minutes).set("content", s.content).set("activity", s.activity));
        at += s.minutes;
      }
      return Value::object()
          .set("designs", designs)
          .set("plan", plan)
          .set("title", "Fighting \"fake news\" (Example 10.2)")
          .set("outcomes", std::vector<std::string>{"Apply criteria for evaluating news stories", "Discuss strategies for responsible sharing"})
          .set("materials", "Laptop and projector, flip chart or whiteboard, \"Sample News Stories\" handout");
    };
    return m;
  }();
  return &model;
}

}  // namespace syn::models
