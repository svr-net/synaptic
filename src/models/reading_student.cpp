// The reader as a student: it learns to read for answers with every learning method the library
// models, and the QA golden test measures what each one adds.
//
// A lesson is a golden question about a FineWeb-Edu passage (other passages than the test's). The
// student answers with one of the passage's sentences or two adjacent ones, then sees the golden
// answer: the reward is the token F1 of its answer. What it learns is how to weigh its reading
// habits — the overlap of a candidate with the question's informative words (BM25), its position,
// whether it is the first sentence, its length, whether it is itself a question, whether it is two
// sentences, whether it holds what the question asks for (a number for "how many", a name for
// "who"), and how much of the question it covers. Agents are students, each with its own order of
// lessons; jobs are study designs: sets of the library's learning methods, each a switch —
//   reinforcement (conditioning, operant): after every answer the habits of the better answers gain
//     weight and those of the worse lose it, in proportion to the reward (the expected-reward
//     gradient of a Rescorla–Wagner learner); without it there is no feedback;
//   schema (schema): it starts from its prior knowledge — its memory — instead of the lexical reader;
//   interleaving (interleaving): question kinds mixed; without it, studied in blocks of one kind;
//   worked examples (gagne, lecture): before answering, it studies the best answer, a small step
//     towards its habits (guidance);
//   spaced retrieval (memory): lessons answered badly come back after a spacing;
//   equilibration (equilibration): a small surprise is assimilated (a half step), a large one
//     accommodated (a double step);
//   scaffolding (zpd, scaffolding): easy lessons before hard ones in the first pass, with support (a
//     larger step) that fades;
//   peer discussion (dialogue, discussion): it compares its answer with a classmate's (the lexical
//     reader's) and, where the classmate's was better, takes a step towards it;
//   self-direction (andragogy): after the first pass it picks, from a few lessons, the one it is
//     least sure about;
//   growth mindset (motivation): it keeps its effort after a run of misses; without it, effort falls
//     with each miss in a row;
//   reflection (chapter 14): every block of lessons it reviews its journal and changes one or two
//     habits at a time.
// The designs: no feedback; its memory alone; reinforcement; reinforcement with each method; all
// methods; all methods but one. After the lessons, and at checkpoints along the way, it answers the
// test questions; the score is the token F1 of its answers against the golden ones.
//
// It starts from its memory (reading_memory.hpp), if the memory was learned for these habits, and
// returns what it would remember next: the habits learned with all methods. With `notes`, it learns
// the other half of its memory, taking notes: from a passage alone, without a question, which
// sentence is worth keeping — the lesson's golden answer says which one was. With `construct`, it
// learns the third, constructing: how to say an answer from the facts it recalls from its knowledge
// capsule (`recalled`, capsuleRecall's result; lessons and questions name their `capsule`) — which
// fact, and whether to say all of it, its predicate, its object or its subject. With `anchor`, each
// lesson also pulls the habits back towards where it started, so that new lessons of another kind do
// not wash out what it knew.
#include <algorithm>
#include <cmath>
#include <map>
#include <numeric>
#include <set>
#include <string>

#include "reading_memory.hpp"
#include "reading_text.hpp"
#include "syn/models/models.hpp"

// Module-scope state and helpers of the student (after the agent interface).
constexpr const char* student_helpers = R"wgsl(
// Constants: P(0) habits F, P(1) lessons, P(2) test questions, P(3) passes, P(4) rate, P(5) margin,
// P(6) spacing, P(7) block, P(8) pool, P(9) support, P(10) questions table (first, count, best),
// P(11) candidates (F habits, F1), P(12) curriculum order of the lessons, P(13) checkpoints,
// P(14) sharpness, P(15..22) its memory, P(23) anchor, P(24..31) the prior (the lexical reader, the
// classmate), P(32) blocked order of the lessons (by question kind), P(33) worked-example step,
// P(34) surprise that calls for accommodation, P(35) discussion step, P(36) discouragement per miss.
var<private> W : array<f32, 8>;
var<private> START : array<f32, 8>;
var<private> JOURNAL : array<f32, 8>;
fn candidate(c : u32) -> u32 { return u32(P(11u)) + c * 9u; }
fn questionRow(q : u32) -> u32 { return u32(P(10u)) + q * 3u; }
fn scoreOf(c : u32) -> f32 {
  let b = candidate(c);
  var s = 0.0;
  for (var k = 0u; k < 8u; k = k + 1u) { s = s + W[k] * P(b + k); }
  return s;
}
// The candidate the student believes answers question q, and its margin over the runner-up.
fn choose(q : u32) -> u32 {
  let r = questionRow(q);
  let first = u32(P(r));
  let count = u32(P(r + 1u));
  var best = first;
  var bestS = -1e30;
  for (var c = first; c < first + count; c = c + 1u) {
    let s = scoreOf(c);
    if (s > bestS) { bestS = s; best = c; }
  }
  return best;
}
fn margin(q : u32) -> f32 {
  let r = questionRow(q);
  let first = u32(P(r));
  let count = u32(P(r + 1u));
  var a = -1e30;
  var b = -1e30;
  for (var c = first; c < first + count; c = c + 1u) {
    let s = scoreOf(c);
    if (s > a) { b = a; a = s; } else if (s > b) { b = s; }
  }
  return a - b;
}
fn rewardOf(c : u32) -> f32 { return P(candidate(c) + 8u); }
// One lesson: answer, see the golden answer — how good every candidate was — and learn: the
// gradient of the expected reward of the student's choices (a softmax over its beliefs, sharpness
// P(14)), so that habits of better answers gain weight and habits of worse ones lose it (or it is
// noted in the journal). Returns the reward its own answer missed.
fn lesson(q : u32, rate : f32, reflect : bool) -> f32 {
  let r = questionRow(q);
  let first = u32(P(r));
  let count = u32(P(r + 1u));
  let beta = P(14u);
  var top = -1e30;
  for (var c = first; c < first + count; c = c + 1u) { top = max(top, scoreOf(c)); }
  var z = 0.0;
  var meanR = 0.0;
  var meanF : array<f32, 8>;
  for (var k = 0u; k < 8u; k = k + 1u) { meanF[k] = 0.0; }
  for (var c = first; c < first + count; c = c + 1u) {
    let w = exp(beta * (scoreOf(c) - top));
    z = z + w;
    meanR = meanR + w * rewardOf(c);
    for (var k = 0u; k < 8u; k = k + 1u) { meanF[k] = meanF[k] + w * P(candidate(c) + k); }
  }
  meanR = meanR / z;
  for (var k = 0u; k < 8u; k = k + 1u) { meanF[k] = meanF[k] / z; }
  var g : array<f32, 8>;
  for (var k = 0u; k < 8u; k = k + 1u) { g[k] = 0.0; }
  for (var c = first; c < first + count; c = c + 1u) {
    let w = exp(beta * (scoreOf(c) - top)) / z * (rewardOf(c) - meanR);
    for (var k = 0u; k < 8u; k = k + 1u) { g[k] = g[k] + w * (P(candidate(c) + k) - meanF[k]); }
  }
  for (var k = 0u; k < 8u; k = k + 1u) {
    let d = rate * beta * g[k];
    if (reflect) {
      JOURNAL[k] = JOURNAL[k] + d;
    } else {
      // The anchor: what it knew holds unless the lessons keep calling for a change.
      W[k] = W[k] + d - rate * P(23u) * (W[k] - START[k]);
    }
  }
  let chosen = choose(q);
  return rewardOf(u32(P(r + 2u))) - rewardOf(chosen);
}
// Reflection: change the one or two habits the journal points at most, then start a new page.
fn reflect() {
  var first = 0u;
  var second = 1u;
  for (var k = 0u; k < 8u; k = k + 1u) {
    if (abs(JOURNAL[k]) > abs(JOURNAL[first])) { second = first; first = k; }
    else if (k != first && abs(JOURNAL[k]) > abs(JOURNAL[second])) { second = k; }
  }
  W[first] = W[first] + JOURNAL[first];
  W[second] = W[second] + JOURNAL[second];
  // The anchor, for the block of lessons the journal covers.
  let hold = pow(max(0.0, 1.0 - P(4u) * P(23u)), P(7u));
  for (var k = 0u; k < 8u; k = k + 1u) {
    JOURNAL[k] = 0.0;
    W[k] = START[k] + (W[k] - START[k]) * hold;
  }
}
// A step from the habits of candidate `own` towards those of candidate `to` (a worked example, or
// a classmate's better answer): into the journal when reflecting.
fn stepTowards(own : u32, to : u32, s : f32, reflect : bool) {
  for (var k = 0u; k < 8u; k = k + 1u) {
    let d = s * (P(candidate(to) + k) - P(candidate(own) + k));
    if (reflect) { JOURNAL[k] = JOURNAL[k] + d; } else { W[k] = W[k] + d; }
  }
}
// The classmate's answer: the candidate the lexical reader (the prior) believes most.
fn classmate(q : u32) -> u32 {
  let r = questionRow(q);
  let first = u32(P(r));
  var best = first;
  var bestS = -1e30;
  for (var c = first; c < first + u32(P(r + 1u)); c = c + 1u) {
    var s = 0.0;
    for (var k = 0u; k < 8u; k = k + 1u) { s = s + P(24u + k) * P(candidate(c) + k); }
    if (s > bestS) { bestS = s; best = c; }
  }
  return best;
}
// The mean reward of the student's answers to questions [start, start + n).
fn exam(start : u32, n : u32) -> f32 {
  var total = 0.0;
  for (var q = start; q < start + n; q = q + 1u) { total = total + rewardOf(choose(q)); }
  return total / f32(max(n, 1u));
}
)wgsl";

constexpr const char* student_agent = R"wgsl(
  // The study design: which of the library's learning methods the student uses (one bit each).
  let methods = u32(J(0u));
  let reinforced = (methods & 1u) != 0u;
  let schema = (methods & 2u) != 0u;
  let interleaved = (methods & 4u) != 0u;
  let examples = (methods & 8u) != 0u;
  let spaced = (methods & 16u) != 0u;
  let equilibrating = (methods & 32u) != 0u;
  let scaffolded = (methods & 64u) != 0u;
  let discussing = (methods & 128u) != 0u;
  let selfDirected = (methods & 256u) != 0u;
  let growth = (methods & 512u) != 0u;
  let reflecting = (methods & 1024u) != 0u;
  let lessons = u32(P(1u));
  let tests = u32(P(2u));
  let passes = u32(P(3u));
  let rate = P(4u);
  // What it knows before the lessons: its memory (a schema), or the lexical reader.
  for (var k = 0u; k < 8u; k = k + 1u) {
    W[k] = select(P(24u + k), P(15u + k), schema);
    START[k] = W[k];
    JOURNAL[k] = 0.0;
  }
  // The order of the lessons: interleaved (its own shuffle), blocked by question kind, or easy
  // before hard (scaffolding).
  var order : array<u32, 512>;
  for (var i = 0u; i < lessons; i = i + 1u) { order[i] = u32(P(u32(P(32u)) + i)); }
  var queue : array<u32, 64>;
  var due : array<u32, 64>;
  for (var i = 0u; i < 64u; i = i + 1u) { queue[i] = 0u; due[i] = 0u; }
  var head = 0u;
  var tail = 0u;
  var done : array<bool, 512>;
  let total = passes * lessons;
  let checkpoints = u32(P(13u));
  var step = 0u;
  var mark = 0u;
  var draw = 100u;
  var misses = 0u;
  for (var sweep = 0u; sweep < passes; sweep = sweep + 1u) {
    for (var i = 0u; i < lessons; i = i + 1u) { done[i] = false; }
    if (scaffolded && sweep == 0u) {
      for (var i = 0u; i < lessons; i = i + 1u) { order[i] = u32(P(u32(P(12u)) + i)); }
    } else if (interleaved) {
      for (var i = lessons - 1u; i > 0u; i = i - 1u) {
        let j = min(i, u32(U(draw) * f32(i + 1u)));
        draw = draw + 1u;
        let t = order[i];
        order[i] = order[j];
        order[j] = t;
      }
    } else {
      for (var i = 0u; i < lessons; i = i + 1u) { order[i] = u32(P(u32(P(32u)) + i)); }
    }
    for (var i = 0u; i < lessons; i = i + 1u) {
      // Checkpoints: answer the test questions now and then.
      if (step * checkpoints >= mark * total && mark < checkpoints) {
        stat(mark, exam(lessons, tests));
        mark = mark + 1u;
      }
      step = step + 1u;
      if (!reinforced) { continue; }
      // Spaced retrieval: a lesson answered badly comes back when it is due.
      if (spaced && head != tail && due[head % 64u] <= step) {
        _ = lesson(queue[head % 64u], rate, reflecting);
        head = head + 1u;
      }
      // The next lesson: in order, or the least certain of a few (self-direction).
      var q = order[i];
      if (selfDirected && sweep > 0u) {
        var bestM = 1e30;
        for (var k = 0u; k < u32(P(8u)); k = k + 1u) {
          let c = order[min(lessons - 1u, i + u32(U(draw) * f32(lessons - i)))];
          draw = draw + 1u;
          if (done[c]) { continue; }
          let m = margin(c);
          if (m < bestM) { bestM = m; q = c; }
        }
      }
      if (done[q]) { q = order[i]; }
      done[q] = true;
      // Effort: scaffolding's support fades over the first pass; without a growth mindset, a run of
      // misses discourages.
      var r = rate;
      if (scaffolded && sweep == 0u) { r = r * (1.0 + P(9u) * (1.0 - f32(i) / f32(lessons))); }
      if (!growth) { r = r / (1.0 + P(36u) * f32(misses)); }
      // A worked example first: a small step towards the best answer's habits.
      let best = u32(P(questionRow(q) + 2u));
      if (examples) { stepTowards(choose(q), best, r * P(33u), reflecting); }
      // Its own answer, and how surprised the feedback leaves it.
      let answer = choose(q);
      let surprise = rewardOf(best) - rewardOf(answer);
      if (equilibrating) { r = r * select(0.5, 2.0, surprise > P(34u)); }
      let missed = lesson(q, r, reflecting);
      // Discussion: where the classmate's answer was better, a step towards it.
      if (discussing) {
        let other = classmate(q);
        if (rewardOf(other) > rewardOf(answer) + P(5u)) { stepTowards(answer, other, r * P(35u), reflecting); }
      }
      misses = select(0u, misses + 1u, surprise > P(5u));
      if (spaced && missed > P(5u) && tail - head < 64u) {
        queue[tail % 64u] = q;
        due[tail % 64u] = step + u32(P(6u));
        tail = tail + 1u;
      }
      if (reflecting && step % u32(P(7u)) == 0u) { reflect(); }
    }
  }
  if (reflecting) { reflect(); }
  stat(16u, exam(lessons, tests));
  stat(17u, exam(0u, lessons));
  for (var k = 0u; k < 8u; k = k + 1u) { stat(18u + k, W[k]); }
  // How it answers the test: two sentences, and the best candidate.
  var pairs = 0.0;
  var right = 0.0;
  for (var q = lessons; q < lessons + tests; q = q + 1u) {
    let c = choose(q);
    pairs = pairs + P(candidate(c) + 5u);
    right = right + select(0.0, 1.0, c == u32(P(questionRow(q) + 2u)));
  }
  stat(26u, pairs / f32(max(tests, 1u)));
  stat(27u, right / f32(max(tests, 1u)));
)wgsl";

namespace syn::models {

namespace {

using namespace text;
using memory::Candidates;
using memory::Item;

// The built-in lessons: golden questions about other passages than the test's.
struct Embedded {
  const char* question;
  const char* answer;
  const char* passage;
};
const Embedded kLessons[] = {
#include "qa_training_sample.inc"
};

// The library's learning methods, one bit each of a study design (reinforcement is bit 0).
struct Method {
  const char* name;
  const char* models;  // the library's models of it
};
const Method kMethods[11] = {{"reinforcement", "conditioning, operant"},
                             {"schema: starts from its memory", "schema"},
                             {"interleaving question kinds", "interleaving"},
                             {"worked examples", "gagne, lecture"},
                             {"spaced retrieval of mistakes", "memory"},
                             {"equilibration: assimilate or accommodate", "equilibration"},
                             {"scaffolding: easy before hard", "zpd, scaffolding"},
                             {"peer discussion", "dialogue, discussion"},
                             {"self-direction", "andragogy"},
                             {"growth mindset", "motivation"},
                             {"reflection: one or two changes at a time", "reflection"}};
constexpr unsigned kAll = (1u << 11) - 1;

// The designs: no feedback, its memory alone, reinforcement, reinforcement with each method, all
// methods, all methods but one.
struct Design {
  unsigned methods;
  std::string name;
};
std::vector<Design> designs() {
  std::vector<Design> d = {{0, "no feedback (the lexical reader)"}, {2, "its memory alone, no feedback"}, {1, "reinforcement"}};
  for (unsigned b = 1; b < 11; ++b) d.push_back({1u | (1u << b), std::string("reinforcement + ") + kMethods[b].name});
  d.push_back({kAll, "all methods"});
  for (unsigned b = 1; b < 11; ++b) d.push_back({kAll & ~(1u << b), std::string("all methods but ") + kMethods[b].name});
  return d;
}
std::size_t designOf(unsigned methods) {
  const auto d = designs();
  for (std::size_t j = 0; j < d.size(); ++j)
    if (d[j].methods == methods) return j;
  return 0;
}

// The kind of a question, for blocked study: its first word.
std::string kindOf(const std::string& question) {
  const auto ws = words(question, false);
  return ws.empty() ? std::string() : ws[0];
}

std::vector<Item> lessonsOf(const Value& v) {
  std::vector<Item> out;
  for (const auto& x : v.asArray()) out.push_back({x.at("question").asString(), x.at("golden").asString(), x.str("passage", "")});
  return out;
}

// The facts each lesson or question may answer from: those recalled from the capsule it names.
std::vector<std::vector<memory::Fact>> capsulesOf(const Value& v, const std::vector<std::vector<memory::Fact>>& recalled) {
  std::vector<std::vector<memory::Fact>> out;
  for (const auto& x : v.asArray()) {
    const double c = x.num("capsule", -1);
    if (c < 0 || c >= static_cast<double>(recalled.size())) throw std::invalid_argument("readingStudent: a lesson names a capsule that is not recalled");
    out.push_back(recalled[static_cast<std::size_t>(c)]);
  }
  return out;
}

double dot(const std::vector<double>& f, const double* w) {
  double s = 0;
  for (int k = 0; k < memory::kHabitCount; ++k) s += f[static_cast<std::size_t>(k)] * w[k];
  return s;
}

}  // namespace

const kernel::Model* readingStudentModel() {
  static const kernel::Model model = [] {
    kernel::Model m;
    m.name = "readingStudent";
    m.title = "The reader as a student: learning to read for answers with every learning method of the library, scored on golden answers";
    m.maxStats = 28;
    m.agentSource = student_agent;
    m.helpersWgsl = student_helpers;
    m.compile = [](const Value& spec) {
      const Value& c = spec.sub("readingStudent");
      std::vector<Item> train, test;
      if (c.has("train")) train = lessonsOf(c.at("train"));
      else
        for (const auto& l : kLessons) train.push_back({l.question, l.answer, l.passage});
      if (c.has("test")) test = lessonsOf(c.at("test"));
      else {
        const Value g = goldenSet(c.integer("questions", 60));
        const auto& ps = g.at("passages").asArray();
        for (const auto& q : g.at("questions").asArray())
          test.push_back({q.at("question").asString(), q.at("golden").asString(), ps[static_cast<std::size_t>(q.at("passage").asNumber())].asString()});
      }
      const auto lessons = std::min<std::size_t>(train.size(), static_cast<std::size_t>(std::clamp(c.integer("lessons", 512), 1, 512)));
      train.resize(lessons);
      const bool notes = c.flag("notes", false);
      const bool construct = c.flag("construct", false);
      std::vector<memory::Candidates> a, b;
      if (construct) {
        if (!c.has("recalled") || !c.has("train") || !c.has("test"))
          throw std::invalid_argument("readingStudent: constructing needs the recalled capsules and the lessons and questions that name them");
        const auto recalled = memory::factsOf(c.at("recalled"));
        auto lessonFacts = capsulesOf(c.at("train"), recalled);
        lessonFacts.resize(train.size());
        a = memory::constructCandidates(train, lessonFacts);
        b = memory::constructCandidates(test, capsulesOf(c.at("test"), recalled));
      } else {
        a = notes ? memory::noteCandidates(train) : memory::answerCandidates(train);
        b = notes ? memory::noteCandidates(test) : memory::answerCandidates(test);
      }
      const double* prior = construct ? memory::kConstructPrior : notes ? memory::kNotePrior : memory::kHabitPrior;
      const char* kind = construct ? "construct" : notes ? "notes" : "answer";
      std::vector<double> params = {8, double(a.size()), double(b.size()), double(std::clamp(c.integer("passes", 3), 1, 10)), c.num("rate", 0.2),
                                    c.num("margin", 0.02), c.num("spacing", 10), double(std::max(1, c.integer("block", 20))), c.num("pool", 6),
                                    c.num("support", 2.0), 0, 0, 0, 16, c.num("sharpness", 5.0)};
      // It starts from its memory — if the memory was learned for these habits — unless asked to start
      // fresh or given the weights to start from.
      const memory::Memory mem = memory::recallFor(c);
      const double* start = construct ? mem.construct : notes ? mem.notes : mem.habits;
      for (int k = 0; k < 8; ++k) params.push_back(start[k]);
      if (c.has("start"))
        for (int k = 0; k < 8 && k < static_cast<int>(c.at("start").size()); ++k) params[15 + k] = c.at("start")[static_cast<std::size_t>(k)].asNumber();
      params.push_back(c.num("anchor", 0.0));
      for (int k = 0; k < 8; ++k) params.push_back(prior[k]);
      params.insert(params.end(), {0, c.num("example", 0.25), c.num("accommodate", 0.3), c.num("discussion", 0.5), c.num("discouragement", 0.1)});
      std::vector<double> questions, candidates;
      Value meta = Value::object();
      double oracle = 0, oracleSingle = 0;
      for (const auto* set : {&a, &b}) {
        for (const auto& p : *set) {
          questions.insert(questions.end(), {double(candidates.size() / 9), double(p.reward.size()), double(candidates.size() / 9 + p.best)});
          for (std::size_t k = 0; k < p.reward.size(); ++k) {
            candidates.insert(candidates.end(), p.habits[k].begin(), p.habits[k].end());
            candidates.push_back(p.reward[k]);
          }
          if (set == &b) {
            oracle += p.reward[p.best];
            double single = 0;
            for (std::size_t k = 0; k < p.reward.size(); ++k)
              if (!p.pair[k]) single = std::max(single, p.reward[k]);
            oracleSingle += single;
          }
        }
      }
      // Easy before hard: lessons ordered by how much the prior misses.
      std::vector<std::size_t> order(a.size());
      std::iota(order.begin(), order.end(), 0);
      auto gap = [&](std::size_t i) {
        const auto& p = a[i];
        std::size_t lex = 0;  // the prior's answer
        for (std::size_t k = 1; k < p.reward.size(); ++k)
          if (dot(p.habits[k], prior) > dot(p.habits[lex], prior)) lex = k;
        return p.reward[p.best] - p.reward[lex];
      };
      std::stable_sort(order.begin(), order.end(), [&](std::size_t x, std::size_t y) { return gap(x) < gap(y); });
      params[10] = double(params.size());
      params.insert(params.end(), questions.begin(), questions.end());
      params[11] = double(params.size());
      params.insert(params.end(), candidates.begin(), candidates.end());
      params[12] = double(params.size());
      for (auto i : order) params.push_back(double(i));
      // Blocked: lessons grouped by question kind (what, how, why, who, …).
      std::vector<std::size_t> blocked(a.size());
      std::iota(blocked.begin(), blocked.end(), 0);
      std::stable_sort(blocked.begin(), blocked.end(), [&](std::size_t x, std::size_t y) { return kindOf(train[x].question) < kindOf(train[y].question); });
      params[32] = double(params.size());
      for (auto i : blocked) params.push_back(double(i));
      meta.set("oracle", oracle / std::max<double>(1, double(b.size()))).set("oracleSingle", oracleSingle / std::max<double>(1, double(b.size())))
          .set("lessons", double(a.size())).set("tests", double(b.size()));
      Value startW = Value::array();
      for (int k = 0; k < 8; ++k) startW.push(params[15 + k]);
      Value recalled = Value::object()
                           .set("used", mem.used && !c.has("start"))
                           .set("reason", c.has("start") ? std::string("started from the given weights") : mem.reason)
                           .set("generation", mem.used ? mem.generation : 0)
                           .set("lessons", mem.used ? mem.lessons : 0.0)
                           .set("design", construct ? mem.constructDesign : notes ? mem.notesDesign : mem.design)
                           .set("source", mem.source)
                           .set("test", construct ? mem.constructTest : notes ? mem.notesTest : mem.test)
                           .set("weights", startW)
                           .set("evidence", mem.used && !c.has("start") ? mem.evidence[construct ? 2 : notes ? 1 : 0] : 0.0)
                           .set("records", mem.records);
      meta.set("source", c.str("source", c.has("train") ? "lessons given to readingStudent" : "the embedded lessons (readingStudent)"));
      meta.set("memory", recalled).set("notes", notes).set("kind", kind);
      std::vector<std::vector<double>> rows;
      for (const auto& d : designs()) rows.push_back({double(d.methods)});
      Plan p = makePlan("readingStudent", static_cast<std::uint32_t>(c.integer("students", 64)), 28, params, rows, seedOf(spec));
      p.meta = meta;
      return std::vector<Plan>{p};
    };
    m.summarise = [](const Value&, const std::vector<Plan>& plans, const std::vector<Output>& out) {
      const Plan& p = plans[0];
      const Output& o = out[0];
      const auto all = designs();
      Value results = Value::array();
      for (std::uint32_t j = 0; j < p.jobs; ++j) {
        Value habits = Value::array();
        const std::string kind = p.meta.at("kind").asString();
        const char* const* names = kind == "construct" ? memory::kConstructHabits : kind == "notes" ? memory::kNoteHabits : memory::kHabits;
        for (int k = 0; k < 8; ++k) habits.push(Value::object().set("habit", names[k]).set("weight", o.mean(p, j, 18 + k)));
        Value used = Value::array();
        for (unsigned b = 0; b < 11; ++b)
          if (all[j].methods & (1u << b)) used.push(kMethods[b].name);
        results.push(Value::object()
                         .set("name", all[j].name)
                         .set("methods", used)
                         .set("curve", Value::Numbers(o.range(p, j, 0, 16)))
                         .set("test", o.mean(p, j, 16))
                         .set("lessons", o.mean(p, j, 17))
                         .set("habits", habits)
                         .set("twoSentences", o.mean(p, j, 26))
                         .set("bestCandidate", o.mean(p, j, 27)));
      }
      // What each method adds to reinforcement alone, and what it costs to leave it out of all.
      const auto& rs = results.asArray();
      auto test = [&](unsigned methods) { return rs[designOf(methods)].num("test", 0); };
      Value methods = Value::array();
      for (unsigned b = 1; b < 11; ++b)
        methods.push(Value::object()
                         .set("method", kMethods[b].name)
                         .set("models", kMethods[b].models)
                         .set("alone", test(1u | (1u << b)) - test(1))
                         .set("removed", test(kAll) - test(kAll & ~(1u << b))));
      // What to remember: the habits it learned with all methods.
      const std::size_t keep = designOf(kAll);
      Value weights = Value::array();
      for (const auto& h : rs[keep].at("habits").asArray()) weights.push(h.at("weight"));
      const Value& recalled = p.meta.at("memory");
      Value next = Value::object()
                       .set("generation", recalled.num("generation", 0) + 1)
                       .set("lessons", recalled.num("lessons", 0) + p.meta.num("lessons", 0) * p.params[3])
                       .set("design", all[keep].name)
                       .set("test", rs[keep].num("test", 0))
                       .set("weights", weights);
      // As a record of the memory: what this study adds, folded in by its evidence. Kept only when it
      // does not answer worse than the memory alone.
      memory::Record rec;
      const std::string kind = p.meta.at("kind").asString();
      rec.section = kind == "construct" ? 2 : kind == "notes" ? 1 : 0;
      rec.generation = static_cast<int>(recalled.num("generation", 0)) + 1;
      rec.lessons = p.meta.num("lessons", 0) * p.params[3];
      rec.test = rs[keep].num("test", 0);
      for (int k = 0; k < memory::kHabitCount; ++k) rec.weights[k] = weights[static_cast<std::size_t>(k)].asNumber();
      rec.design = all[keep].name;
      rec.source = p.meta.at("source").asString();
      double before[memory::kHabitCount], after[memory::kHabitCount];
      for (int k = 0; k < memory::kHabitCount; ++k) before[k] = recalled.at("weights")[static_cast<std::size_t>(k)].asNumber();
      memory::fold(before, recalled.num("evidence", 0), rec, after);
      const double alone = rs[designOf(2)].num("test", 0);
      next.set("record", memory::packRecord(rec))
          .set("folded", Value::Numbers(std::vector<double>(after, after + memory::kHabitCount)))
          .set("evidence", recalled.num("evidence", 0) + rec.lessons)
          .set("accepted", rec.test >= alone - 1e-9)
          .set("reason", rec.test >= alone - 1e-9 ? std::string("it answers at least as well as its memory alone")
                                                   : std::string("it answers worse than its memory alone: not kept"));
      return Value::object().set("designs", results).set("methods", methods).set("memory", recalled).set("remember", next).set("notes", p.meta.at("notes")).set("kind", p.meta.at("kind"))
          .set("fingerprint", memory::fingerprint()).set("oracle", p.meta.at("oracle")).set("oracleSingle", p.meta.at("oracleSingle"))
          .set("lessons", p.meta.at("lessons")).set("tests", p.meta.at("tests")).set("students", static_cast<double>(p.agents));
    };
    return m;
  }();
  return &model;
}

}  // namespace syn::models
