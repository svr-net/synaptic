// Lectures, chunking and attention; instructional video length; storytelling and analogies
// (chapter 10) in the space.
//
// A learner's attention is a point of the affective channel. While they only listen it turns
// away from the learner (a lowering mixture each minute, faster in front of a screen); a
// bridging activity between chunks (a pause, a question, a short demonstration) raises it again
// and lets them retrieve what they heard, which deepens the trace. Each minute of content
// teaches with the chance of their attention times the chance that they follow it (their
// ability against its difficulty, dilated when an analogy connects it to everyday experience),
// as a mixture of their knowledge towards the learner. A week later the knowledge is read
// through the forgetting rotor of the memory model; a story deepens the trace. An online video
// loses each viewer whose attention no longer holds a uniform draw.
// Jobs: a 50-minute session in chunks of 8–25 minutes (or none), with a story or an analogy;
// videos of 2–20 minutes. Agents: learners.
#include <string>

#include "syn/models/models.hpp"

// The agent program, in WGSL.
constexpr const char* lecture_agent = R"wgsl(
  let video = J(0u) > f32(0.5);
  let chunk = J(1u);
  let story = J(2u) > f32(0.5);
  let analogy = J(3u) > f32(0.5);
  let total = u32(J(4u));
  let bridge = P(0u);
  let decay = select(P(1u), P(2u), video);
  let rate = P(4u);
  var attention = sampleShare(U(0u), P(8u), P(9u), kB);
  let ability = sampleLevel(U(1u), f32(0), P(10u), kA);
  // How well they follow: ability against difficulty, dilated by an analogy (most for novices).
  let connect = select(f32(0), P(6u) * (f32(1) - chance(ability)), analogy);
  let follow = chance(shift(atLogit(logit(ability) - P(5u), kA), connect));
  var known = atChance(f32(0), kA);
  var stability = P(14u) * select(f32(1), f32(1) + P(16u), story);
  var sinceBreak = f32(0);
  var bridgeLeft = f32(0);
  var content = f32(0);
  var watching = true;
  var watched = f32(0);
  for (var m = 0u; m < total; m = m + 1u) {
    if (video && watching && U(10u + m) < P(13u) * (f32(1) - chance(attention))) { watching = false; }
    if (!watching) { continue; }
    stat(m, chance(attention));
    if (video && m < 20u) { stat(60u + m, f32(1)); }
    watched = watched + f32(1);
    if (!video && chunk > f32(0.5) && bridgeLeft < f32(0.5) && sinceBreak >= chunk) { bridgeLeft = bridge; sinceBreak = f32(0); }
    if (bridgeLeft > f32(0.5)) {
      // A bridging activity: attention comes back; retrieving what was heard deepens the trace.
      attention = raise(attention, P(3u));
      stability = stability * (f32(1) + P(15u) * chance(known));
      bridgeLeft = bridgeLeft - f32(1);
      continue;
    }
    known = raise(known, rate * chance(attention) * follow);
    attention = lower(attention, decay);
    content = content + f32(1);
    sinceBreak = sinceBreak + f32(1);
  }
  // A week later: the forgetting rotor turns the trace away from the learner.
  let r = f32(1) / sqrt(f32(1) + f32(19) / f32(81) * P(17u) / stability);
  let trace = rotate(pole(), kN, kA, f32(2) * acos(sqrt(r)));
  stat(80u, chance(known));
  stat(81u, chance(known) * chance(trace));
  stat(82u, content);
  stat(83u, select(f32(0), f32(1), watching));
  stat(84u, watched);
)wgsl";

namespace syn::models {

const kernel::Model* lectureModel() {
  static const kernel::Model model = [] {
    kernel::Model m;
    m.name = "lecture";
    m.title = "Lectures: attention and chunking, bridging activities, stories and analogies; instructional video length";
    m.maxStats = 85;
    m.agentSource = lecture_agent;
    m.compile = [](const Value& spec) {
      const Value& c = spec.sub("lecture");
      const double session = std::max(10, std::min(60, c.integer("minutes", 50)));
      const std::vector<double> params = {c.num("bridgeMinutes", 2.0), c.num("decay", 0.035), c.num("screenDecay", 0.07), c.num("restore", 0.45),
                                          c.num("rate", 0.06), c.num("difficulty", 0.5), c.num("analogy", 1.2), 0.0, c.num("attentionMean", 0.85),
                                          c.num("attentionSd", 0.08), c.num("abilitySd", 1.0), 0.0, 0.0, c.num("dropout", 0.3),
                                          c.num("stability", 2.0), c.num("retrieval", 0.25), c.num("story", 0.8), c.num("delayDays", 7.0)};
      std::vector<std::vector<double>> rows;
      for (double chunk : {0.0, 25.0, 15.0, 12.0, 8.0}) rows.push_back({0, chunk, 0, 0, session});
      rows.push_back({0, 12, 1, 0, session});
      rows.push_back({0, 12, 0, 1, session});
      rows.push_back({0, 12, 1, 1, session});
      for (double v : {2.0, 4.0, 6.0, 8.0, 10.0, 15.0, 20.0}) rows.push_back({1, 0, 0, 0, v});
      return std::vector<Plan>{makePlan("lecture", learnersOf(spec, 500), 85, params, rows, seedOf(spec))};
    };
    m.summarise = [](const Value&, const std::vector<Plan>& plans, const std::vector<Output>& out) {
      const Plan& p = plans[0];
      const Output& o = out[0];
      Value sessions = Value::array(), videos = Value::array();
      for (std::uint32_t j = 0; j < p.jobs; ++j) {
        const float* row = p.jobRows.data() + static_cast<std::size_t>(j) * p.jobStride;
        const auto minutes = static_cast<std::uint32_t>(row[4]);
        if (row[0] < 0.5f) {
          std::string name = row[1] < 0.5f ? "no breaks" : "chunks of " + std::to_string(static_cast<int>(row[1])) + " min";
          if (row[2] > 0.5f) name += " + story";
          if (row[3] > 0.5f) name += " + analogy";
          sessions.push(Value::object()
                            .set("name", name)
                            .set("chunk", static_cast<double>(row[1]))
                            .set("story", row[2] > 0.5f)
                            .set("analogy", row[3] > 0.5f)
                            .set("attention", Value::Numbers(o.range(p, j, 0, minutes)))
                            .set("learned", o.mean(p, j, 80))
                            .set("retained", o.mean(p, j, 81))
                            .set("contentMinutes", o.mean(p, j, 82)));
        } else {
          std::vector<double> still;
          for (std::uint32_t k = 0; k < std::min(minutes, 20u); ++k) still.push_back(o.mean(p, j, 60 + k));
          videos.push(Value::object()
                          .set("minutes", static_cast<double>(minutes))
                          .set("watching", Value::Numbers(still))
                          .set("completed", o.mean(p, j, 83))
                          .set("watchedMinutes", o.mean(p, j, 84))
                          .set("learned", o.mean(p, j, 80)));
        }
      }
      return Value::object().set("sessions", sessions).set("videos", videos).set("bridgeMinutes", static_cast<double>(p.params[0]));
    };
    return m;
  }();
  return &model;
}

}  // namespace syn::models
