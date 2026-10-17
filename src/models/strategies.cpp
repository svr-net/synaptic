// Selecting instructional strategies (chapter 10) in the space: "learning styles", content and
// variety; the flipped one-shot; problem-based learning.
//
// Learning styles: every learner states a preferred mode — visual, verbal or hands-on — and
// some learners cannot use one mode at all (a disability, a language barrier: Universal Design
// for Learning). How well a mode teaches depends on the content (a dilation of the learner's
// task by the fit of mode and content: a map is spatial, a database search procedural, "primary
// source" a concept), not on the stated preference, which only moves satisfaction (Pashler et
// al., 2008). A lesson in one mode moves knowledge towards the learner by the chance of the
// dilated task; a variety of modes reaches each learner through their best accessible mode, at a
// cost in time.
// The flipped one-shot: preparation (a video before class) is a cap test of a point that
// accountability (a short questionnaire) and embedding in the course (the faculty assigns it)
// dilate. In class, application builds higher-order skill at a rate set by the basics a
// learner brings; a traditional session spends most of its time telling the basics.
// Problem-based learning: an ill-defined question in place of a lecture moves the same basics,
// raises interest (a mixture towards the learner) and transfer, and takes more time when done in
// full. Agents: learners.
#include <string>

#include "syn/models/models.hpp"

// The agent program, in WGSL.
constexpr const char* strategies_agent = R"wgsl(
  let kind = u32(J(0u));
  let ability = sampleLevel(U(0u), f32(0), P(0u), kA);
  let base = atLogit(logit(ability) - P(1u), kA);
  if (kind == 0u) {
    let content = u32(J(1u));
    let rule = u32(J(2u));
    // A stated preference (a third each) and, for some learners, a mode they cannot use.
    let style = min(2u, u32(U(1u) * f32(3)));
    let barred = inside(atChance(P(2u), kC), chanceCap(U(2u)));
    let lost = min(2u, u32(U(3u) * f32(3)));
    var known = atChance(f32(0), kA);
    var satisfied = atChance(P(3u), kB);
    var lo = 0u;
    var hi = 1u;
    var only = 0u;
    if (rule == 0u) { only = 1u; }                     // the same lecture for everyone (verbal)
    if (rule == 1u) { only = style; }                  // matched to the stated style
    if (rule == 2u) { only = u32(J(3u)); }             // matched to the content
    if (rule == 3u) { lo = 0u; hi = 3u; }              // a variety of modes
    if (rule != 3u) { lo = only; hi = only + 1u; }
    // One mode teaches by the chance of the task dilated by its fit to the content. A variety of
    // modes lets every learner reach the content through the mode that works best for them among
    // those they can use, at the cost of the time each mode takes.
    let cost = select(f32(1), P(6u), hi - lo > 1u);
    var best = f32(0);
    var pleased = f32(0);
    for (var mode = lo; mode < hi; mode = mode + 1u) {
      let fit = J(4u + mode) + select(f32(0), -P(4u), barred && lost == mode);
      best = max(best, chance(shift(base, fit)));
      // A preferred mode pleases; it does not teach more.
      if (mode == style) { pleased = P(5u) / f32(hi - lo); }
    }
    known = raise(known, cost * best);
    satisfied = raise(satisfied, pleased);
    stat(0u, chance(known));
    stat(1u, chance(satisfied));
    stat(2u, select(f32(0), chance(known), barred));
    stat(3u, select(f32(0), f32(1), barred));
    return;
  }
  if (kind == 1u) {
    let flipped = J(1u) > f32(0.5);
    let accountable = J(2u) > f32(0.5);
    let embedded = J(3u) > f32(0.5);
    let session = P(10u);
    // Preparation: a cap test of a point dilated by accountability and embedding.
    let readiness = shift(atChance(P(11u), kB), select(f32(0), P(12u), accountable) + select(f32(0), P(13u), embedded));
    let prepared = flipped && inside(readiness, chanceCap(U(4u)));
    var basics = atChance(f32(0), kA);
    var higher = atChance(f32(0), kA);
    var telling = P(14u);
    if (flipped) {
      telling = P(15u);
      if (prepared) { basics = raise(basics, P(16u) * chance(base)); }
    }
    // Telling the basics in class, then applying them.
    basics = raise(basics, telling / session * P(17u) * chance(base));
    let applying = session - telling;
    for (var m = 0u; m < u32(applying); m = m + 1u) {
      higher = raise(higher, P(18u) * chance(basics) * chance(shift(base, -P(19u))));
    }
    stat(4u, select(f32(0), f32(1), prepared));
    stat(5u, chance(basics));
    stat(6u, chance(higher));
    stat(7u, select(chance(higher), f32(0), prepared));
    stat(8u, select(f32(1), f32(0), prepared));
    return;
  }
  // Problem-based learning: 0 a lecture, 1 elements of PBL in a one-shot, 2 full PBL (three sessions).
  let pbl = u32(J(1u));
  let sessions = select(f32(1), P(20u), pbl == 2u);
  var basics = atChance(f32(0), kA);
  var interest = sampleShare(U(5u), P(21u), f32(0.15), kB);
  var transfer = atChance(f32(0), kA);
  for (var s = 0u; s < u32(sessions); s = s + 1u) {
    basics = raise(basics, P(22u) * chance(base));
    if (pbl > 0u) {
      interest = raise(interest, P(23u));
      transfer = raise(transfer, P(24u) * chance(interest) * chance(base));
    } else {
      transfer = raise(transfer, P(25u) * chance(base));
    }
  }
  stat(9u, chance(basics));
  stat(10u, chance(transfer));
  stat(11u, chance(interest));
  stat(12u, sessions);
)wgsl";

namespace syn::models {

namespace {
const char* kContents[3] = {"spatial (a map of the library)", "procedural (a database search)", "conceptual (what a primary source is)"};
const char* kRules[4] = {"the same lecture for everyone", "matched to each learner's stated style", "matched to the content", "a variety of modes (UDL)"};
// Fit of the visual, verbal and hands-on modes to each kind of content (logits).
const double kFit[3][3] = {{1.0, -0.6, 0.3}, {0.2, -0.4, 1.0}, {0.3, 0.6, -0.1}};
}  // namespace

const kernel::Model* strategiesModel() {
  static const kernel::Model model = [] {
    kernel::Model m;
    m.name = "strategies";
    m.title = "Selecting strategies: learning styles vs content vs variety (UDL); the flipped one-shot; problem-based learning";
    m.maxStats = 13;
    m.agentSource = strategies_agent;
    m.compile = [](const Value& spec) {
      const Value& c = spec.sub("strategies");
      const std::vector<double> params = {c.num("abilitySd", 1.0), c.num("difficulty", 0.0), c.num("barred", 0.1), c.num("satisfaction", 0.5),
                                          c.num("barrier", 3.0), c.num("preference", 0.4), c.num("varietyCost", 0.9), 0, 0, 0,
                                          c.num("sessionMinutes", 50.0), c.num("prepares", 0.35), c.num("accountability", 1.0), c.num("embedding", 1.5),
                                          c.num("tellingMinutes", 30.0), c.num("recapMinutes", 5.0), c.num("videoLearning", 0.85), c.num("telling", 1.1),
                                          c.num("applicationRate", 0.06), c.num("higherOrder", 0.5), c.num("pblSessions", 3.0), c.num("interest", 0.4),
                                          c.num("basics", 0.6), c.num("pblInterest", 0.3), c.num("pblTransfer", 0.45), c.num("lectureTransfer", 0.2)};
      std::vector<std::vector<double>> rows;
      for (int content = 0; content < 3; ++content) {
        int best = 0;
        for (int mode = 1; mode < 3; ++mode)
          if (kFit[content][mode] > kFit[content][best]) best = mode;
        for (int rule = 0; rule < 4; ++rule) rows.push_back({0, double(content), double(rule), double(best), kFit[content][0], kFit[content][1], kFit[content][2]});
      }
      for (const auto& f : std::vector<std::vector<double>>{{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {1, 0, 1}, {1, 1, 1}}) rows.push_back({1, f[0], f[1], f[2], 0, 0, 0});
      for (double pbl : {0.0, 1.0, 2.0}) rows.push_back({2, pbl, 0, 0, 0, 0, 0});
      return std::vector<Plan>{makePlan("strategies", learnersOf(spec, 500), 13, params, rows, seedOf(spec))};
    };
    m.summarise = [](const Value&, const std::vector<Plan>& plans, const std::vector<Output>& out) {
      const Plan& p = plans[0];
      const Output& o = out[0];
      Value styles = Value::array();
      for (int content = 0; content < 3; ++content) {
        Value rules = Value::array();
        for (int rule = 0; rule < 4; ++rule) {
          const auto j = static_cast<std::uint32_t>(content * 4 + rule);
          const double barred = o.mean(p, j, 3);
          rules.push(Value::object()
                         .set("rule", kRules[rule])
                         .set("knowledge", o.mean(p, j, 0))
                         .set("satisfaction", o.mean(p, j, 1))
                         .set("barredKnowledge", barred > 0 ? o.mean(p, j, 2) / barred : 0.0));
        }
        styles.push(Value::object().set("content", kContents[content]).set("rules", rules));
      }
      const char* flippedNames[] = {"traditional one-shot", "flipped", "flipped + pre-work check", "flipped, embedded in the course",
                                    "flipped, embedded + pre-work check"};
      Value flipped = Value::array();
      for (std::uint32_t j = 12; j < 17; ++j) {
        const double unprepared = o.mean(p, j, 8);
        flipped.push(Value::object()
                         .set("name", flippedNames[j - 12])
                         .set("prepared", o.mean(p, j, 4))
                         .set("basics", o.mean(p, j, 5))
                         .set("higherOrder", o.mean(p, j, 6))
                         .set("unpreparedHigherOrder", unprepared > 0 ? o.mean(p, j, 7) / unprepared : 0.0));
      }
      const char* pblNames[] = {"lecture", "PBL elements in a one-shot", "full problem-based learning"};
      Value pbl = Value::array();
      for (std::uint32_t j = 17; j < 20; ++j)
        pbl.push(Value::object()
                     .set("name", pblNames[j - 17])
                     .set("basics", o.mean(p, j, 9))
                     .set("transfer", o.mean(p, j, 10))
                     .set("interest", o.mean(p, j, 11))
                     .set("sessions", o.mean(p, j, 12)));
      return Value::object()
          .set("modes", std::vector<std::string>{"visual", "verbal", "hands-on"})
          .set("styles", styles)
          .set("flipped", flipped)
          .set("pbl", pbl);
    };
    return m;
  }();
  return &model;
}

}  // namespace syn::models
