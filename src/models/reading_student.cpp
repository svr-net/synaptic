// The reader as a student: it learns to read for answers with the library's own modules, in the
// order of the book, and the QA golden test measures what each one adds.
//
// A lesson is a golden question about a FineWeb-Edu passage (other passages than the test's). The
// student answers with one of the passage's sentences or two adjacent ones, then sees the golden
// answer: the reward is the token F1 of its answer, and the lesson is the candidate it should have
// chosen (the one closest to the golden answer). What it learns is how to weigh its reading
// habits — the overlap of a candidate with the question's informative words (BM25), its position,
// whether it is the first sentence, its length, whether it is itself a question, whether it is two
// sentences, whether it holds what the question asks for (a number for "how many", a name for
// "who"), and how much of the question it covers. It starts as the lexical reader of readingQA
// (overlap only, one sentence). Agents are students, each with its own order of lessons; jobs are
// study designs, each adding one module to the one before:
//   0 no feedback — the lexical reader;
//   1 behaviourism — reinforcement: after every answer the weights move towards the habits of the
//     better answer, in proportion to the reward it missed (a Rescorla–Wagner/perceptron step);
//   2 + cognitivism — spaced retrieval: lessons answered badly come back after a spacing;
//   3 + social constructivism — scaffolding in the ZPD: easy lessons before hard ones in the first
//     pass, with support (a large step) that fades;
//   4 + andragogy — self-direction: the student picks, from a few lessons, the one it is least sure
//     about;
//   5 + reflection — every block of lessons the student reviews what went wrong and changes one or
//     two habits at a time (chapter 14).
// After the lessons, and at checkpoints along the way, it answers the test questions; the score is
// the token F1 of its answers against the golden ones.
#include <algorithm>
#include <cmath>
#include <map>
#include <numeric>
#include <set>
#include <string>

#include "reading_text.hpp"
#include "syn/models/models.hpp"

// Module-scope state and helpers of the student (after the agent interface).
constexpr const char* student_helpers = R"wgsl(
// Constants: P(0) habits F, P(1) lessons, P(2) test questions, P(3) passes, P(4) rate, P(5) margin,
// P(6) spacing, P(7) block, P(8) pool, P(9) support, P(10) questions table (first, count, best),
// P(11) candidates (F habits, F1), P(12) curriculum order of the lessons, P(13) checkpoints.
var<private> W : array<f32, 8>;
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
    if (reflect) { JOURNAL[k] = JOURNAL[k] + d; } else { W[k] = W[k] + d; }
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
  for (var k = 0u; k < 8u; k = k + 1u) { JOURNAL[k] = 0.0; }
}
// The mean reward of the student's answers to questions [start, start + n).
fn exam(start : u32, n : u32) -> f32 {
  var total = 0.0;
  for (var q = start; q < start + n; q = q + 1u) { total = total + rewardOf(choose(q)); }
  return total / f32(max(n, 1u));
}
)wgsl";

constexpr const char* student_agent = R"wgsl(
  let design = u32(J(0u));
  let lessons = u32(P(1u));
  let tests = u32(P(2u));
  let passes = u32(P(3u));
  let rate = P(4u);
  // Prior knowledge: the lexical reader (overlap with the question; one sentence).
  for (var k = 0u; k < 8u; k = k + 1u) {
    W[k] = 0.0;
    JOURNAL[k] = 0.0;
  }
  W[0] = 1.0;
  W[5] = -1.0;
  // The order of the lessons: a shuffle of the student's own, or easy before hard (scaffolding).
  var order : array<u32, 512>;
  for (var i = 0u; i < lessons; i = i + 1u) { order[i] = i; }
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
  for (var sweep = 0u; sweep < passes; sweep = sweep + 1u) {
    for (var i = 0u; i < lessons; i = i + 1u) { done[i] = false; }
    if (design >= 3u && sweep == 0u) {
      for (var i = 0u; i < lessons; i = i + 1u) { order[i] = u32(P(u32(P(12u)) + i)); }
    } else {
      for (var i = lessons - 1u; i > 0u; i = i - 1u) {
        let j = min(i, u32(U(draw) * f32(i + 1u)));
        draw = draw + 1u;
        let t = order[i];
        order[i] = order[j];
        order[j] = t;
      }
    }
    for (var i = 0u; i < lessons; i = i + 1u) {
      // Checkpoints: answer the test questions now and then.
      if (step * checkpoints >= mark * total && mark < checkpoints) {
        stat(mark, exam(lessons, tests));
        mark = mark + 1u;
      }
      step = step + 1u;
      if (design == 0u) { continue; }
      // Spaced retrieval: a lesson answered badly comes back when it is due.
      if (design >= 2u && head != tail && due[head % 64u] <= step) {
        _ = lesson(queue[head % 64u], rate, design >= 5u);
        head = head + 1u;
      }
      // The next lesson: in order, or the least certain of a few (self-direction).
      var q = order[i];
      if (design >= 4u && sweep > 0u) {
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
      // Scaffolding: a large step at first, fading over the first pass.
      var r = rate;
      if (design >= 3u && sweep == 0u) { r = rate * (1.0 + P(9u) * (1.0 - f32(i) / f32(lessons))); }
      let missed = lesson(q, r, design >= 5u);
      if (design >= 2u && missed > P(5u) && tail - head < 64u) {
        queue[tail % 64u] = q;
        due[tail % 64u] = step + u32(P(6u));
        tail = tail + 1u;
      }
      if (design >= 5u && step % u32(P(7u)) == 0u) { reflect(); }
    }
  }
  if (design >= 5u) { reflect(); }
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

struct Lesson {
  std::string question, golden, passage;
};

// The built-in lessons: golden questions about other passages than the test's.
struct Embedded {
  const char* question;
  const char* answer;
  const char* passage;
};
const Embedded kLessons[] = {
#include "qa_training_sample.inc"
};

const char* kHabits[8] = {"overlap with the question (BM25)", "position in the passage", "first sentence", "length",
                          "is itself a question", "two sentences", "holds what is asked (a number, a name)", "covers the question"};
const char* kDesigns[6] = {"no feedback (the lexical reader)", "behaviourism: reinforcement from feedback", "+ cognitivism: spaced retrieval of mistakes",
                           "+ social constructivism: scaffolding, easy before hard", "+ andragogy: self-directed study",
                           "+ reflection: one or two changes at a time"};

std::vector<Lesson> lessonsOf(const Value& v) {
  std::vector<Lesson> out;
  for (const auto& x : v.asArray()) out.push_back({x.at("question").asString(), x.at("golden").asString(), x.at("passage").asString()});
  return out;
}

bool hasDigit(const std::string& s) { return std::any_of(s.begin(), s.end(), [](unsigned char c) { return std::isdigit(c) != 0; }); }
bool hasName(const std::string& s) {
  for (std::size_t i = 1; i + 1 < s.size(); ++i)
    if (s[i - 1] == ' ' && std::isupper(static_cast<unsigned char>(s[i])) && std::islower(static_cast<unsigned char>(s[i + 1]))) return true;
  return false;
}

// Candidates of one question (its passage's sentences, then adjacent pairs) with their habits and
// the token F1 of each against the golden answer.
struct Prepared {
  std::vector<std::vector<double>> habits;
  std::vector<double> reward;
  std::size_t best = 0;
};

std::vector<Prepared> prepare(const std::vector<Lesson>& items) {
  // What the reader learns from the passages it reads: how informative each word is.
  std::map<std::string, double> df;
  std::vector<std::vector<std::string>> sents(items.size());
  double totalLen = 0, nSent = 0;
  for (std::size_t i = 0; i < items.size(); ++i) {
    sents[i] = sentences(items[i].passage);
    std::set<std::string> seen;
    for (const auto& s : sents[i]) {
      const auto ws = words(s, true);
      totalLen += static_cast<double>(ws.size());
      nSent += 1;
      seen.insert(ws.begin(), ws.end());
    }
    for (const auto& w : seen) df[w] += 1;
  }
  const double n = static_cast<double>(items.size()), avg = totalLen / std::max(1.0, nSent);
  auto idf = [&](const std::string& w) {
    const double d = df.count(w) ? df[w] : 0.0;
    return std::max(0.0, std::log((n - d + 0.5) / (d + 0.5) + 1.0));
  };
  std::vector<Prepared> out;
  for (std::size_t i = 0; i < items.size(); ++i) {
    const auto& q = items[i].question;
    std::set<std::string> qw;
    for (const auto& w : words(q, true)) qw.insert(w);
    std::string lower = q;
    std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    const bool asksNumber = lower.find("how many") != std::string::npos || lower.find("how much") != std::string::npos ||
                            lower.find("percent") != std::string::npos || lower.find("what year") != std::string::npos ||
                            lower.rfind("when", 0) == 0 || lower.find("how long") != std::string::npos || lower.find("how old") != std::string::npos;
    const bool asksName = lower.rfind("who", 0) == 0 || lower.find(" who ") != std::string::npos || lower.find("which person") != std::string::npos;
    const auto& ss = sents[i];
    std::vector<std::string> texts;
    std::vector<int> pos;
    std::vector<bool> pair;
    for (std::size_t k = 0; k < ss.size() && texts.size() < 40; ++k) {
      texts.push_back(ss[k]);
      pos.push_back(static_cast<int>(k));
      pair.push_back(false);
    }
    for (std::size_t k = 0; k + 1 < ss.size() && texts.size() < 64; ++k) {
      texts.push_back(ss[k] + " " + ss[k + 1]);
      pos.push_back(static_cast<int>(k));
      pair.push_back(true);
    }
    Prepared p;
    double maxBm = 1e-9;
    std::vector<double> bm(texts.size());
    for (std::size_t c = 0; c < texts.size(); ++c) {
      const auto ws = words(texts[c], true);
      std::map<std::string, double> tf;
      for (const auto& w : ws) tf[w] += 1;
      const double norm = 1.2 * (1 - 0.75 + 0.75 * static_cast<double>(ws.size()) / avg);
      double e = 0;
      for (const auto& w : qw)
        if (tf.count(w)) e += idf(w) * tf[w] * 2.2 / (tf[w] + norm);
      bm[c] = e;
      if (!pair[c]) maxBm = std::max(maxBm, e);
    }
    for (std::size_t c = 0; c < texts.size(); ++c) {
      const auto ws = words(texts[c], true);
      std::set<std::string> have(ws.begin(), ws.end());
      double covered = 0;
      for (const auto& w : qw) covered += have.count(w) ? 1.0 : 0.0;
      const double fit = (asksNumber && hasDigit(texts[c])) || (asksName && hasName(texts[c])) ? 1.0 : 0.0;
      p.habits.push_back({bm[c] / maxBm, ss.size() > 1 ? double(pos[c]) / double(ss.size() - 1) : 0.0, pos[c] == 0 && !pair[c] ? 1.0 : 0.0,
                          static_cast<double>(words(texts[c], false).size()) / 40.0, texts[c].find('?') != std::string::npos ? 1.0 : 0.0,
                          pair[c] ? 1.0 : 0.0, fit, qw.empty() ? 0.0 : covered / static_cast<double>(qw.size())});
      p.reward.push_back(tokenF1(texts[c], items[i].golden));
    }
    for (std::size_t c = 1; c < p.reward.size(); ++c)
      if (p.reward[c] > p.reward[p.best]) p.best = c;
    out.push_back(std::move(p));
  }
  return out;
}

}  // namespace

const kernel::Model* readingStudentModel() {
  static const kernel::Model model = [] {
    kernel::Model m;
    m.name = "readingStudent";
    m.title = "The reader as a student: learning to read for answers with the library's modules, scored on golden answers";
    m.maxStats = 28;
    m.agentSource = student_agent;
    m.helpersWgsl = student_helpers;
    m.compile = [](const Value& spec) {
      const Value& c = spec.sub("readingStudent");
      std::vector<Lesson> train, test;
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
      const auto a = prepare(train), b = prepare(test);
      std::vector<double> params = {8, double(a.size()), double(b.size()), double(std::clamp(c.integer("passes", 3), 1, 10)), c.num("rate", 0.2),
                                    c.num("margin", 0.02), c.num("spacing", 10), double(std::max(1, c.integer("block", 20))), c.num("pool", 6),
                                    c.num("support", 2.0), 0, 0, 0, 16, c.num("sharpness", 5.0)};
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
              if (p.habits[k][5] < 0.5) single = std::max(single, p.reward[k]);
            oracleSingle += single;
          }
        }
      }
      // Easy before hard: lessons ordered by how much the lexical reader misses.
      std::vector<std::size_t> order(a.size());
      std::iota(order.begin(), order.end(), 0);
      auto gap = [&](std::size_t i) {
        const auto& p = a[i];
        std::size_t lex = 0;  // the lexical reader's answer: the single sentence of most overlap
        for (std::size_t k = 1; k < p.reward.size(); ++k)
          if (p.habits[k][5] < 0.5 && p.habits[k][0] > p.habits[lex][0]) lex = k;
        return p.reward[p.best] - p.reward[lex];
      };
      std::stable_sort(order.begin(), order.end(), [&](std::size_t x, std::size_t y) { return gap(x) < gap(y); });
      params[10] = double(params.size());
      params.insert(params.end(), questions.begin(), questions.end());
      params[11] = double(params.size());
      params.insert(params.end(), candidates.begin(), candidates.end());
      params[12] = double(params.size());
      for (auto i : order) params.push_back(double(i));
      meta.set("oracle", oracle / std::max<double>(1, double(b.size()))).set("oracleSingle", oracleSingle / std::max<double>(1, double(b.size())))
          .set("lessons", double(a.size())).set("tests", double(b.size()));
      std::vector<std::vector<double>> rows;
      for (int d = 0; d < 6; ++d) rows.push_back({double(d)});
      Plan p = makePlan("readingStudent", static_cast<std::uint32_t>(c.integer("students", 64)), 28, params, rows, seedOf(spec));
      p.meta = meta;
      return std::vector<Plan>{p};
    };
    m.summarise = [](const Value&, const std::vector<Plan>& plans, const std::vector<Output>& out) {
      const Plan& p = plans[0];
      const Output& o = out[0];
      Value designs = Value::array();
      for (std::uint32_t j = 0; j < p.jobs; ++j) {
        Value habits = Value::array();
        for (int k = 0; k < 8; ++k) habits.push(Value::object().set("habit", kHabits[k]).set("weight", o.mean(p, j, 18 + k)));
        designs.push(Value::object()
                         .set("name", kDesigns[j])
                         .set("curve", Value::Numbers(o.range(p, j, 0, 16)))
                         .set("test", o.mean(p, j, 16))
                         .set("lessons", o.mean(p, j, 17))
                         .set("habits", habits)
                         .set("twoSentences", o.mean(p, j, 26))
                         .set("bestCandidate", o.mean(p, j, 27)));
      }
      return Value::object().set("designs", designs).set("oracle", p.meta.at("oracle")).set("oracleSingle", p.meta.at("oracleSingle"))
          .set("lessons", p.meta.at("lessons")).set("tests", p.meta.at("tests")).set("students", static_cast<double>(p.agents));
    };
    return m;
  }();
  return &model;
}

}  // namespace syn::models
