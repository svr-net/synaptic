// Reflective practice (chapter 14) in the space: a term of teaching, and what each way of
// reflecting on it changes.
//
// An agent is an instructor teaching a term of sessions. Their delivery is seven points of the
// cognitive channel — the seven items of the peer observation form (Example 14.1): voice, pace,
// engagement with the content, eye contact, moving away from the podium, use of the space,
// presence. Some aspects are blind spots (playing with a bracelet, a filler "so"): the instructor
// cannot see them from inside and believes they are fine. What a session teaches is the mean
// chance of the seven. After a session the instructor reflects, perceiving each aspect with an
// error that depends on the evidence: informal reflection (feelings and anecdotes) is noisy,
// worse the longer they wait, and is seldom applied; the four-step process (York-Barr et al.:
// what? why? so what? now what?) in a journal describes before it judges, and ends in an
// action; beside the journal, a video recording, a peer observation (a trusted colleague's
// Fair/Good/Excellent, some harsher than others) and student feedback (Brookfield's critical
// incident questionnaire) see the blind spots too. The instructor then works on the aspects that look weakest: a change of a
// habit sticks with the chance of the effort it gets (a cap test), so one or two at a time stick
// and seven at once do not. A negativity bias (Brookfield) colours what they remember: harsh
// reflection drains self-efficacy, a point of the affective channel, and a drained instructor
// skips reflecting; celebrating successes restores it. Jobs: no reflection, informal, the
// four-step journal, the journal with video, peer observation, student feedback or all three;
// and the journal changing one or every aspect, written a week later, or harsh.
#include <string>

#include "syn/models/models.hpp"

// The agent program, in WGSL.
constexpr const char* reflection_agent = R"wgsl(
  let method = u32(J(0u));
  let focus = u32(J(1u));
  let delay = J(2u);
  let kind = J(3u) > f32(0.5);
  let sessions = u32(P(0u));
  var aspect : array<V5, 7>;
  var blind : array<bool, 7>;
  for (var k = 0u; k < 7u; k = k + 1u) {
    blind[k] = inside(atChance(P(1u), kC), chanceCap(U(k)));
    aspect[k] = sampleLevel(U(10u + k), P(2u) - select(f32(0), P(3u), blind[k]), P(4u), kA);
  }
  var efficacy = atChance(P(5u), kB);
  // A peer observer's harshness ("why didn't I think of that?").
  let harsh = P(23u) * kLogisticUnit * logisticOf(U(20u));
  var hits = f32(0);
  var choices = f32(0);
  var reflections = f32(0);
  var skipped = f32(0);
  for (var s = 0u; s < sessions; s = s + 1u) {
    var quality = f32(0);
    for (var k = 0u; k < 7u; k = k + 1u) { quality = quality + chance(aspect[k]); }
    quality = quality / f32(7);
    stat(s, quality);
    // Experience alone: a little better every session.
    for (var k = 0u; k < 7u; k = k + 1u) { aspect[k] = shift(aspect[k], P(6u)); }
    if (method == 0u) { continue; }
    let base = 100u + s * 64u;
    // The evidence beside the journal: student feedback every session, a video every fifth, a peer every tenth.
    let video = (method == 3u || method == 6u) && s % 5u == 4u;
    let peer = (method == 4u || method == 6u) && s % 10u == 9u;
    let ciq = method == 5u || method == 6u;
    // A drained instructor does not reflect.
    if (!inside(efficacy, chanceCap(P(7u) * U(base)))) {
      skipped = skipped + f32(1);
      continue;
    }
    reflections = reflections + f32(1);
    // Perceiving each aspect: the error of the evidence, growing with the days since the session.
    var noise = select(P(9u), P(8u), method == 1u) * (f32(1) + P(10u) * delay);
    if (ciq) { noise = P(11u); }
    if (video) { noise = P(12u); }
    var seen : array<f32, 7>;
    var felt = f32(0);
    for (var k = 0u; k < 7u; k = k + 1u) {
      let q = chance(aspect[k]);
      var p = q + noise * kLogisticUnit * logisticOf(U(base + 1u + k));
      if (peer) { p = (floor(clamp(q - harsh + P(13u) * kLogisticUnit * logisticOf(U(base + 8u + k)), f32(0), f32(0.999)) * f32(3)) + f32(0.5)) / f32(3); }
      let sees = video || peer || ciq;
      if (blind[k] && !sees) { p = P(14u); }
      // Negative moments dominate memory, more so after a delay.
      p = p - P(15u) * (f32(1) + delay / f32(7)) * select(f32(0), f32(1), p < P(16u));
      seen[k] = p;
      felt = felt + p / f32(7);
    }
    // Self-efficacy: harsh reflection drains it; being kind and celebrating successes restores it.
    efficacy = lower(efficacy, P(17u) * (f32(1) - felt));
    if (kind) { efficacy = raise(efficacy, P(18u) * felt); }
    // Informal reflection is rarely turned into a change of practice; "now what?" is.
    if (method == 1u && !inside(atChance(P(19u), kC), chanceCap(U(base + 20u)))) { continue; }
    // Work on the aspects that look weakest; a habit changes with the chance of its share of the effort.
    var picked : array<bool, 7>;
    for (var k = 0u; k < 7u; k = k + 1u) { picked[k] = false; }  // explicitly, every session
    for (var r = 0u; r < focus; r = r + 1u) {
      var worst = 99u;
      for (var k = 0u; k < 7u; k = k + 1u) {
        if (!picked[k] && (worst == 99u || seen[k] < seen[worst])) { worst = k; }
      }
      picked[worst] = true;
      // Was it really among the two weakest?
      var below = 0u;
      for (var k = 0u; k < 7u; k = k + 1u) {
        if (k != worst && chance(aspect[k]) < chance(aspect[worst])) { below = below + 1u; }
      }
      hits = hits + select(f32(0), f32(1), below < 2u);
      choices = choices + f32(1);
      let sticks = inside(atLogit(P(20u) / f32(focus) - P(21u), kA), chanceCap(U(base + 30u + r)));
      if (sticks) { aspect[worst] = shift(aspect[worst], P(22u)); }
    }
  }
  var quality = f32(0);
  var blindQuality = f32(0);
  var blindCount = f32(0);
  for (var k = 0u; k < 7u; k = k + 1u) {
    quality = quality + chance(aspect[k]) / f32(7);
    if (blind[k]) {
      blindQuality = blindQuality + chance(aspect[k]);
      blindCount = blindCount + f32(1);
    }
  }
  stat(40u, quality);
  stat(41u, chance(efficacy));
  stat(42u, blindQuality);
  stat(43u, blindCount);
  stat(44u, hits);
  stat(45u, choices);
  stat(46u, reflections);
  stat(47u, skipped);
)wgsl";

namespace syn::models {

namespace {
const char* kMethods[7] = {"no reflection", "informal (feelings, anecdotes)", "four-step reflective journal", "journal + video (every fifth session)",
                           "journal + peer observation (every tenth session)", "journal + student feedback (CIQ)", "journal + all three"};
}  // namespace

const kernel::Model* reflectionModel() {
  static const kernel::Model model = [] {
    kernel::Model m;
    m.name = "reflection";
    m.title = "Reflective practice: informal reflection, the four-step journal, video, peer observation and student feedback over a term";
    m.maxStats = 48;
    m.agentSource = reflection_agent;
    m.compile = [](const Value& spec) {
      const Value& c = spec.sub("reflection");
      const int sessions = std::max(5, std::min(40, c.integer("sessions", 30)));
      const std::vector<double> params = {double(sessions), c.num("blindSpots", 0.3), c.num("skill", 0.6), c.num("blindGap", 0.8),
                                          c.num("skillSd", 0.5), c.num("efficacy", 0.7), c.num("experience", 0.004), c.num("drain", 0.5),
                                          c.num("informalNoise", 0.25), c.num("journalNoise", 0.1), c.num("delayNoise", 0.3),
                                          c.num("feedbackNoise", 0.08), c.num("videoNoise", 0.03), c.num("peerNoise", 0.08),
                                          c.num("belief", 0.8), c.num("negativity", 0.06), c.num("negativeBelow", 0.6),
                                          c.num("harshness", 0.25), c.num("celebrate", 0.25), c.num("informalApplied", 0.3),
                                          c.num("habit", 7.0), c.num("habitThreshold", 3.6), c.num("change", 0.35), c.num("observerSd", 0.1)};
      std::vector<std::vector<double>> rows;
      for (int method = 0; method < 7; ++method) rows.push_back({double(method), 2, 0, 1});
      rows.push_back({2, 1, 0, 1});
      rows.push_back({2, 7, 0, 1});
      rows.push_back({2, 2, 7, 1});
      rows.push_back({2, 2, 0, 0});
      return std::vector<Plan>{makePlan("reflection", static_cast<std::uint32_t>(c.integer("instructors", 1000)), 48, params, rows, seedOf(spec))};
    };
    m.summarise = [](const Value&, const std::vector<Plan>& plans, const std::vector<Output>& out) {
      const Plan& p = plans[0];
      const Output& o = out[0];
      const auto sessions = static_cast<std::uint32_t>(p.params[0]);
      const char* variants[] = {"journal, one change at a time", "journal, changing everything at once", "journal, a week later", "journal, harsh on yourself"};
      Value designs = Value::array();
      for (std::uint32_t j = 0; j < p.jobs; ++j) {
        const double blind = o.mean(p, j, 43), choices = o.mean(p, j, 45);
        designs.push(Value::object()
                         .set("name", j < 7 ? kMethods[j] : variants[j - 7])
                         .set("quality", Value::Numbers(o.range(p, j, 0, sessions)))
                         .set("final", o.mean(p, j, 40))
                         .set("gain", o.mean(p, j, 40) - o.mean(p, j, 0))
                         .set("efficacy", o.mean(p, j, 41))
                         .set("blindSpots", blind > 0 ? o.mean(p, j, 42) / blind : 0.0)
                         .set("rightTarget", choices > 0 ? o.mean(p, j, 44) / choices : 0.0)
                         .set("reflections", o.mean(p, j, 46))
                         .set("skipped", o.mean(p, j, 47)));
      }
      return Value::object()
          .set("designs", designs)
          .set("sessions", static_cast<double>(sessions))
          .set("aspects", std::vector<std::string>{"voice", "pace", "engagement with the content", "eye contact", "moving away from the podium",
                                                   "use of the space", "presence"})
          .set("steps", std::vector<std::string>{"What happened? (description)", "Why? (analysis and interpretation)", "So what? (meaning and application)",
                                                 "Now what? (implications for action)"});
    };
    return m;
  }();
  return &model;
}

}  // namespace syn::models
