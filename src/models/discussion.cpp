// Discussion (chapter 10) in the space: who speaks, and what everyone learns.
//
// An agent is a class. Each learner has a confidence point (affective channel), a knowledge point
// (cognitive channel) and the seconds they need to formulate an answer — more for English
// language learners, neurodiverse learners and learners with learning disabilities. After a
// question, a learner is ready when the silence the instructor leaves holds their time (a cap of
// the logistic of the difference), and willing to speak to the whole class when their
// confidence, dilated by the risk of speaking in that setting, holds a uniform draw. The
// instructor calls on raised hands, the most confident first. Speaking (articulating) and
// listening both move a learner's knowledge towards the learner, open questions more than
// closed ones. Formats: the whole class; think-pair-share (everyone rehearses with a partner,
// which raises confidence, before sharing); small groups (everyone speaks in the group, one
// reporter each); a fishbowl (eight speak, the rest listen actively); a circle of voices (every
// learner a turn). Dominant talkers are managed with three chips each or by waiting until two
// other voices have spoken. Jobs: those designs and a sweep of the wait time.
#include <string>

#include "syn/models/models.hpp"

// The agent program, in WGSL.
constexpr const char* discussion_agent = R"wgsl(
  let format = u32(J(0u));
  let wait = J(1u);
  let manage = u32(J(2u));
  let open = J(3u) > f32(0.5);
  let n = u32(P(0u));
  let questions = u32(P(1u));
  let turns = u32(P(2u));
  let gain = select(f32(1), P(13u), open);
  // Speaking to the whole class is riskier for open questions; less so after rehearsing.
  let risk = -P(11u) - select(f32(0), P(16u), open);
  var confidence : array<V5, 32>;
  var known : array<V5, 32>;
  var needs : array<f32, 32>;
  var flagged : array<bool, 32>;
  var spoke : array<u32, 32>;
  var articulated : array<bool, 32>;
  var waitFor : array<u32, 32>;
  var start = f32(0);
  var flaggedCount = f32(0);
  for (var i = 0u; i < n; i = i + 1u) {
    confidence[i] = sampleShare(U(i * 8u), P(3u), P(4u), kB);
    flagged[i] = inside(atChance(P(5u), kC), chanceCap(U(i * 8u + 1u)));
    needs[i] = P(6u) * exp(P(7u) * kLogisticUnit * logisticOf(U(i * 8u + 2u))) * select(f32(1), P(8u), flagged[i]);
    known[i] = sampleShare(U(i * 8u + 3u), P(9u), P(10u), kA);
    start = start + chance(known[i]);
    flaggedCount = flaggedCount + select(f32(0), f32(1), flagged[i]);
  }
  var silences = f32(0);
  var total = f32(0);
  for (var q = 0u; q < questions; q = q + 1u) {
    let base = 1000u + q * 4096u;
    var shift5 = f32(0);
    if (format == 1u) {
      // Think-pair-share: everyone rehearses an answer with a partner first.
      for (var i = 0u; i < n; i = i + 1u) {
        known[i] = raise(known[i], P(14u) * gain);
        confidence[i] = raise(confidence[i], P(12u));
        articulated[i] = true;
      }
      shift5 = P(17u);
    }
    if (format == 2u) {
      // Small groups of four: everyone speaks in the group; one reporter per group shares.
      for (var i = 0u; i < n; i = i + 1u) {
        known[i] = raise(known[i], P(14u) * gain);
        articulated[i] = true;
      }
      for (var g = 0u; g * 4u < n; g = g + 1u) {
        var best = g * 4u;
        for (var i = g * 4u; i < min(n, g * 4u + 4u); i = i + 1u) {
          if (chance(confidence[i]) > chance(confidence[best])) { best = i; }
        }
        spoke[best] = spoke[best] + 1u;
        total = total + f32(1);
        for (var i = 0u; i < n; i = i + 1u) { known[i] = raise(known[i], P(15u) * gain); }
      }
      continue;
    }
    if (format == 4u) {
      // A circle of voices: every learner speaks in turn, uninterrupted.
      if (q * 2u >= questions) { continue; }
      for (var i = 0u; i < n; i = i + 1u) {
        spoke[i] = spoke[i] + 1u;
        articulated[i] = true;
        total = total + f32(1);
        known[i] = raise(known[i], P(14u) * gain);
        for (var k = 0u; k < n; k = k + 1u) { known[k] = raise(known[k], P(15u) * gain * f32(0.5)); }
      }
      continue;
    }
    for (var t = 0u; t < turns; t = t + 1u) {
      // Raised hands: ready (the silence holds their time) and willing (confidence vs risk).
      var chosen = 99u;
      var bestHand = f32(-1);
      for (var i = 0u; i < n; i = i + 1u) {
        if (format == 3u && i >= 8u) { continue; }  // fishbowl: the inner circle speaks
        if (manage == 1u && spoke[i] >= 3u) { continue; }
        if (manage == 2u && waitFor[i] > 0u) { continue; }
        let k = base + (t * 32u + i) * 4u;
        let ready = inside(atLogit((wait - needs[i]) / P(18u), kA), chanceCap(U(k)));
        let willing = inside(shift(confidence[i], risk + shift5 + select(f32(0), P(17u), format == 3u)), chanceCap(U(k + 1u)));
        let hand = chance(confidence[i]) + f32(0.2) * U(k + 2u);
        if (ready && willing && hand > bestHand) { bestHand = hand; chosen = i; }
      }
      if (chosen == 99u) {
        silences = silences + f32(1);
        continue;
      }
      spoke[chosen] = spoke[chosen] + 1u;
      articulated[chosen] = true;
      total = total + f32(1);
      known[chosen] = raise(known[chosen], P(14u) * gain);
      confidence[chosen] = raise(confidence[chosen], P(19u));
      for (var i = 0u; i < n; i = i + 1u) {
        if (waitFor[i] > 0u) { waitFor[i] = waitFor[i] - 1u; }
        let listen = select(f32(1), P(20u), format == 3u && i >= 8u);
        known[i] = raise(known[i], P(15u) * gain * listen);
      }
      if (manage == 2u) { waitFor[chosen] = 2u; }
    }
  }
  // Who spoke, and how concentrated the talk was (the five most frequent speakers).
  var voices = f32(0);
  var flaggedVoices = f32(0);
  var anyVoice = f32(0);
  var end = f32(0);
  var taken : array<bool, 32>;
  var top = f32(0);
  for (var r = 0u; r < 5u; r = r + 1u) {
    var best = 99u;
    for (var i = 0u; i < n; i = i + 1u) {
      if (!taken[i] && (best == 99u || spoke[i] > spoke[best])) { best = i; }
    }
    if (best != 99u) {
      taken[best] = true;
      top = top + f32(spoke[best]);
    }
  }
  for (var i = 0u; i < n; i = i + 1u) {
    voices = voices + select(f32(0), f32(1), spoke[i] > 0u);
    flaggedVoices = flaggedVoices + select(f32(0), f32(1), spoke[i] > 0u && flagged[i]);
    anyVoice = anyVoice + select(f32(0), f32(1), articulated[i]);
    end = end + chance(known[i]);
  }
  stat(0u, voices / f32(n));
  stat(1u, anyVoice / f32(n));
  stat(2u, select(f32(0), top / total, total > f32(0)));
  stat(3u, flaggedVoices);
  stat(4u, flaggedCount);
  stat(5u, end / f32(n));
  stat(6u, (end - start) / f32(n));
  stat(7u, silences);
  stat(8u, total);
)wgsl";

namespace syn::models {

namespace {
const char* kFormats[5] = {"whole class", "think-pair-share", "small groups", "fishbowl", "circle of voices"};
const char* kManage[3] = {"", " + three chips each", " + wait for two other voices"};
}  // namespace

const kernel::Model* discussionModel() {
  static const kernel::Model model = [] {
    kernel::Model m;
    m.name = "discussion";
    m.title = "Discussion: wait time, think-pair-share and groups, managing dominant talkers, open and closed questions";
    m.maxStats = 9;
    m.agentSource = discussion_agent;
    m.compile = [](const Value& spec) {
      const Value& c = spec.sub("discussion");
      const int n = std::max(4, std::min(32, c.integer("classSize", 24)));
      const std::vector<double> params = {double(n), double(std::max(1, std::min(16, c.integer("questions", 8)))), double(c.integer("turnsPerQuestion", 3)),
                                          c.num("confidenceMean", 0.4), c.num("confidenceSd", 0.2), c.num("needsMoreTime", 0.2), c.num("seconds", 6.0),
                                          c.num("secondsSd", 0.5), c.num("moreTimeFactor", 2.0), c.num("knownMean", 0.4), c.num("knownSd", 0.15),
                                          c.num("risk", 1.5), c.num("rehearsal", 0.25), c.num("openGain", 1.6), c.num("speakGain", 0.12),
                                          c.num("listenGain", 0.015), c.num("openRisk", 0.4), c.num("smallRisk", 1.0), c.num("readySeconds", 2.5),
                                          c.num("speakingConfidence", 0.1), c.num("activeListening", 1.6)};
      std::vector<std::vector<double>> rows;
      const double w = c.num("wait", 15.0);
      for (int f = 0; f < 5; ++f) rows.push_back({double(f), w, 0, 1});
      rows.push_back({0, w, 1, 1});
      rows.push_back({0, w, 2, 1});
      rows.push_back({0, w, 0, 0});
      for (double s : {1.0, 3.0, 5.0, 10.0, 15.0, 20.0, 30.0}) rows.push_back({0, s, 0, 1});
      return std::vector<Plan>{makePlan("discussion", static_cast<std::uint32_t>(c.integer("classes", 400)), 9, params, rows, seedOf(spec))};
    };
    m.summarise = [](const Value&, const std::vector<Plan>& plans, const std::vector<Output>& out) {
      const Plan& p = plans[0];
      const Output& o = out[0];
      auto row = [&](std::uint32_t j) {
        const float* r = p.jobRows.data() + static_cast<std::size_t>(j) * p.jobStride;
        const double flagged = o.mean(p, j, 4);
        return Value::object()
            .set("wait", static_cast<double>(r[1]))
            .set("open", r[3] > 0.5f)
            .set("voices", o.mean(p, j, 0))
            .set("articulated", o.mean(p, j, 1))
            .set("topFiveShare", o.mean(p, j, 2))
            .set("moreTimeVoices", flagged > 0 ? o.mean(p, j, 3) / flagged : 0.0)
            .set("learned", o.mean(p, j, 6))
            .set("knowledge", o.mean(p, j, 5))
            .set("silences", o.mean(p, j, 7))
            .set("turns", o.mean(p, j, 8));
      };
      Value designs = Value::array(), waits = Value::array();
      for (std::uint32_t j = 0; j < 8; ++j) {
        const float* r = p.jobRows.data() + static_cast<std::size_t>(j) * p.jobStride;
        std::string name = std::string(kFormats[static_cast<int>(r[0])]) + kManage[static_cast<int>(r[2])];
        if (r[3] < 0.5f) name += ", closed questions";
        designs.push(row(j).set("name", name));
      }
      for (std::uint32_t j = 8; j < p.jobs; ++j) waits.push(row(j));
      return Value::object().set("designs", designs).set("waitTime", waits).set("classSize", static_cast<double>(p.params[0]));
    };
    return m;
  }();
  return &model;
}

}  // namespace syn::models
