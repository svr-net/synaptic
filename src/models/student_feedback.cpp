// Student feedback (chapter 14) in the space: reading patterns, not single comments; ratings
// that do not measure learning and are biased.
//
// An agent is one class session of five segments, taught to n learners. Each segment has a
// clarity point of the cognitive channel; with some probability one of them is confusing. A
// learner understands a segment when its clarity, dilated by their ability, holds a uniform draw
// (a cap test). After the session they answer Brookfield's critical incident questionnaire: the
// moment they found most puzzling — honestly when it is anonymous, less often when it is not —
// and, now and then, a grumble about a segment that was fine. They also rate the session (1–5):
// a level that follows how much they liked it more than how much they learned, and sits lower
// for an instructor who is a woman or a person of colour with the same teaching (Buskist and
// Hogan; Owen). The instructor revises the lesson by one of three rules: every segment anyone
// complained about; only the segments a share of the class named (a pattern); or, from the
// rating alone, a segment at random when the mean is low. A revised confusing segment becomes
// clear; revising a segment that worked costs a little. Jobs: the three rules, a named
// questionnaire, the two identities, and a sweep of the pattern threshold.
#include <string>

#include "syn/models/models.hpp"

// The agent program, in WGSL.
constexpr const char* feedback_agent = R"wgsl(
  let rule = u32(J(0u));
  let anonymous = J(1u) > f32(0.5);
  let biased = J(2u) > f32(0.5);
  let threshold = J(3u);
  let n = u32(P(0u));
  let confusing = select(99u, min(4u, u32(U(1u) * f32(5))), inside(atChance(P(1u), kC), chanceCap(U(0u))));
  var clarity : array<V5, 5>;
  for (var k = 0u; k < 5u; k = k + 1u) {
    clarity[k] = atChance(select(P(2u), P(3u), k == confusing), kA);
  }
  var mentions : array<f32, 5>;
  var learned = f32(0);
  var rating = f32(0);
  var ratingLearning = f32(0);
  var ratingSq = f32(0);
  var learningSq = f32(0);
  for (var i = 0u; i < n; i = i + 1u) {
    let base = 10u + i * 16u;
    let ability = P(4u) * kLogisticUnit * logisticOf(U(base));
    var understood = f32(0);
    var puzzled = 99u;
    for (var k = 0u; k < 5u; k = k + 1u) {
      if (inside(shift(clarity[k], ability), chanceCap(U(base + 1u + k)))) {
        understood = understood + f32(1);
      } else if (puzzled == 99u || k == confusing) {
        puzzled = k;
      }
    }
    // The most puzzling moment, honestly when anonymous; or a grumble about any segment.
    let honest = select(P(6u), P(5u), anonymous);
    if (puzzled != 99u && U(base + 6u) < honest) { mentions[puzzled] = mentions[puzzled] + f32(1); }
    if (U(base + 7u) < P(7u)) {
      let g = min(4u, u32(U(base + 8u) * f32(5)));
      mentions[g] = mentions[g] + f32(1);
    }
    let l = understood / f32(5);
    // The rating: liking more than learning, and a bias that has nothing to do with the teaching.
    let level = sampleLevel(U(base + 9u), P(8u) + P(9u) * (l - f32(0.5)) - select(f32(0), P(10u), biased), P(11u), kB);
    let r = f32(1) + f32(4) * chance(level);
    learned = learned + l;
    rating = rating + r;
    ratingLearning = ratingLearning + r * l;
    ratingSq = ratingSq + r * r;
    learningSq = learningSq + l * l;
  }
  learned = learned / f32(n);
  rating = rating / f32(n);
  // The revision.
  var fixed = false;
  var falseRevisions = f32(0);
  var revised : array<bool, 5>;
  for (var k = 0u; k < 5u; k = k + 1u) {
    if (rule == 0u) { revised[k] = mentions[k] > f32(0.5); }
    if (rule == 1u) { revised[k] = mentions[k] / f32(n) >= threshold; }
  }
  if (rule == 2u && rating < P(12u)) { revised[min(4u, u32(U(2u) * f32(5)))] = true; }
  var after = f32(0);
  for (var k = 0u; k < 5u; k = k + 1u) {
    var c = clarity[k];
    if (revised[k]) {
      if (k == confusing) {
        fixed = true;
        c = atChance(P(2u), kA);
      } else {
        falseRevisions = falseRevisions + f32(1);
        c = shift(c, -P(13u));
      }
    }
    after = after + chance(c) / f32(5);
  }
  stat(0u, select(f32(0), f32(1), confusing != 99u));
  stat(1u, select(f32(0), f32(1), fixed));
  stat(2u, falseRevisions);
  stat(3u, learned);
  stat(4u, after);
  stat(5u, rating);
  stat(6u, ratingLearning / f32(n));
  stat(7u, ratingSq / f32(n));
  stat(8u, learningSq / f32(n));
  stat(9u, select(f32(0), mentions[min(confusing, 4u)] / f32(n), confusing != 99u));
  var stray = f32(0);
  for (var k = 0u; k < 5u; k = k + 1u) {
    if (k != confusing && mentions[k] > f32(0.5)) { stray = stray + f32(1); }
  }
  stat(10u, stray);
  // Before: the clarity of the segments as taught.
  var before = f32(0);
  for (var k = 0u; k < 5u; k = k + 1u) { before = before + chance(clarity[k]) / f32(5); }
  stat(11u, before);
)wgsl";

namespace syn::models {

const kernel::Model* studentFeedbackModel() {
  static const kernel::Model model = [] {
    kernel::Model m;
    m.name = "studentFeedback";
    m.title = "Student feedback: the critical incident questionnaire, patterns vs single comments, ratings and their bias";
    m.maxStats = 12;
    m.agentSource = feedback_agent;
    m.compile = [](const Value& spec) {
      const Value& c = spec.sub("studentFeedback");
      const int n = std::max(4, std::min(60, c.integer("classSize", 25)));
      const double pattern = c.num("pattern", 0.2);
      const std::vector<double> params = {double(n), c.num("confusing", 0.5), c.num("clear", 0.88), c.num("unclear", 0.45),
                                          c.num("abilitySd", 1.0), c.num("anonymousHonesty", 0.9), c.num("namedHonesty", 0.45),
                                          c.num("grumble", 0.06), c.num("liking", 0.9), c.num("learningWeight", 0.6), c.num("bias", 0.35),
                                          c.num("ratingSd", 0.8), c.num("ratingThreshold", 3.85), c.num("revisionCost", 0.4)};
      std::vector<std::vector<double>> rows = {{0, 1, 0, pattern}, {1, 1, 0, pattern}, {2, 1, 0, pattern}, {1, 0, 0, pattern}, {1, 1, 1, pattern}, {2, 1, 1, pattern}};
      for (double t : {0.04, 0.08, 0.12, 0.16, 0.2, 0.3, 0.4, 0.5}) rows.push_back({1, 1, 0, t});
      return std::vector<Plan>{makePlan("studentFeedback", static_cast<std::uint32_t>(c.integer("sessions", 2000)), 12, params, rows, seedOf(spec))};
    };
    m.summarise = [](const Value&, const std::vector<Plan>& plans, const std::vector<Output>& out) {
      const Plan& p = plans[0];
      const Output& o = out[0];
      auto row = [&](std::uint32_t j) {
        const float* r = p.jobRows.data() + static_cast<std::size_t>(j) * p.jobStride;
        const double confusing = o.mean(p, j, 0);
        // The correlation of a learner's rating with what they learned, pooled over the sessions.
        const double er = o.mean(p, j, 5), el = o.mean(p, j, 3);
        const double cov = o.mean(p, j, 6) - er * el, vr = o.mean(p, j, 7) - er * er, vl = o.mean(p, j, 8) - el * el;
        return Value::object()
            .set("threshold", static_cast<double>(r[3]))
            .set("anonymous", r[1] > 0.5f)
            .set("biased", r[2] > 0.5f)
            .set("caught", confusing > 0 ? o.mean(p, j, 1) / confusing : 0.0)
            .set("falseRevisions", o.mean(p, j, 2))
            .set("learned", o.mean(p, j, 3))
            .set("before", o.mean(p, j, 11))
            .set("after", o.mean(p, j, 4))
            .set("rating", er)
            .set("ratingLearning", vr > 0 && vl > 0 ? cov / std::sqrt(vr * vl) : 0.0)
            .set("confusingShare", confusing > 0 ? o.mean(p, j, 9) / confusing : 0.0)
            .set("strayComments", o.mean(p, j, 10));
      };
      const char* names[] = {"revise whatever anyone complained about", "revise the patterns", "act on the ratings",
                             "patterns, questionnaire with names", "patterns, instructor rated with bias", "ratings, instructor rated with bias"};
      Value rules = Value::array(), sweep = Value::array();
      for (std::uint32_t j = 0; j < 6; ++j) rules.push(row(j).set("name", names[j]));
      for (std::uint32_t j = 6; j < p.jobs; ++j) sweep.push(row(j));
      return Value::object()
          .set("rules", rules)
          .set("threshold", sweep)
          .set("classSize", static_cast<double>(p.params[0]))
          .set("ciq", std::vector<std::string>{"At what moment in class this week were you most engaged as a learner?",
                                               "At what moment in class this week were you most distanced as a learner?",
                                               "What action that anyone in the room took did you find most affirming or helpful?",
                                               "What action that anyone in the room took did you find most puzzling or confusing?",
                                               "What about the class this week surprised you the most?"});
    };
    return m;
  }();
  return &model;
}

}  // namespace syn::models
