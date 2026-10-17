// Learning meaning in the space (functional geometric algebra, Pustejovsky 2026, §11).
//
// An agent is a learner who reads a short text — sentences "Ann is older than Ben.", "Fay is a
// sibling of Ben.", "Ann is not a sibling of Ben." — and learns a lexicon: one multivector of
// Cl(4, 1) per word, truncated to grades 0–2 (§11.1: a scalar, a vector and a bivector part,
// 16 coefficients). The meaning of "s r o" is the scalar part of the geometric product,
// ⟨S R O⟩₀ — for an entity vector, a relation bivector and an entity vector exactly the paper's
// Layer 1, s ⌋ (r ⌊ o) — and the learner's belief that it is true is that scalar read as a point
// of the space: chance(atLogit(⟨S R O⟩₀)). Learning is gradient descent on the coefficients
// (§11.5: the product is multilinear, so its gradient is the product of the other two), with the
// contrastive objective of §11.6: each sentence is learned along with corrupted ones (a random
// subject or object), which count as false unless the text says otherwise.
// Learners differ in what the algebra imposes (§11.3, §11.7): typed (entities are unit vectors of
// V_E = ⟨A, B, C⟩, relations bivectors), typed with a scalar part in the relation, untyped
// multivectors, untyped with a grade-consistency prior, and a flat learner that composes the same
// 16 numbers coefficient by coefficient (DistMult: the algebra's multiplication table ignored).
// Every learner starts from its own random lexicon; the text is the same. With the default text
// (a family of six, "older than" and "sibling of"), the world is known, so the model also asks what
// the learners infer about sentences they never read: the reverse of a sentence (asymmetry), the
// transitive closure of "older than", and the converse of "sibling of" (symmetry).
#include <algorithm>
#include <array>
#include <cctype>
#include <sstream>
#include <string>

#include "syn/core/multivector.hpp"
#include "syn/models/models.hpp"
#include "syn/space/algebra.hpp"

// The learner's lexicon and its operations (module scope, after the algebra).
constexpr const char* learning_helpers = R"wgsl(
// ---- a lexicon of multivectors truncated to grades 0–2 (16 coefficients per word) ----
// Constants: Q(0..9) sizes and rates; Q(16 + 4f) the sentences (s, r, o, truth); Q(280 + t) the
// sentences stated true; Q(480 + t) the world's truth; Q(680 + t) the test set of t; Q(880 + 4k)
// the terms (a, b, c, τ) of ⟨S R O⟩₀ = Σ τ S_a R_b O_c, in three lists (typed, typed with a scalar,
// untyped) starting at term Q(10 + list), Q(13 + list) terms each.
var<private> W : array<f32, 176>;
var<private> G : array<f32, 48>;
var<private> MASK_E : array<f32, 16>;
var<private> MASK_R : array<f32, 16>;
var<private> OFF_E : array<f32, 16>;
var<private> OFF_R : array<f32, 16>;
// ⟨S R O⟩₀ for the words s, r, o (r counts from the first relation word), and its gradient in G.
fn meaning(s : u32, r : u32, o : u32, kind : u32) -> f32 {
  let ws = s * 16u;
  let wr = (u32(Q(0u)) + r) * 16u;
  let wo = o * 16u;
  for (var i = 0u; i < 48u; i = i + 1u) { G[i] = 0.0; }
  var total = 0.0;
  if (kind == 4u) {
    // Flat: the same 16 numbers, multiplied coefficient by coefficient.
    for (var i = 0u; i < 16u; i = i + 1u) {
      total = total + W[ws + i] * W[wr + i] * W[wo + i];
      G[i] = W[wr + i] * W[wo + i];
      G[16u + i] = W[ws + i] * W[wo + i];
      G[32u + i] = W[ws + i] * W[wr + i];
    }
    return total;
  }
  // The terms of the product that the learner's blades can reach (one list per kind of learner).
  let list = min(kind, 2u);
  let first = u32(Q(10u + list));
  let terms = u32(Q(13u + list));
  for (var k = first; k < first + terms; k = k + 1u) {
    let a = u32(Q(880u + 4u * k));
    let b = u32(Q(881u + 4u * k));
    let c = u32(Q(882u + 4u * k));
    let tau = Q(883u + 4u * k);
    let x = W[ws + a];
    let y = W[wr + b];
    let z = W[wo + c];
    total = total + tau * x * y * z;
    G[a] = G[a] + tau * y * z;
    G[16u + b] = G[16u + b] + tau * x * z;
    G[32u + c] = G[32u + c] + tau * x * y;
  }
  return total;
}
// The belief that the sentence is true: its meaning read as a point of the space.
fn belief(m : f32) -> f32 { return chance(atLogit(clamp(m, -30.0, 30.0), kA)); }
// A judgement: right (1), wrong (0), or undecided (½) when the belief is ½ — as it is exactly for
// "x r x" with a bivector relation, where the device's rounding should not decide.
fn judged(p : f32, truth : f32) -> f32 {
  if (abs(p - 0.5) < 1e-3) { return 0.5; }
  return select(0.0, 1.0, (p > 0.5) == (truth > 0.5));
}
// Entities are unit-normalised (§5.2): their coefficients have size 1.
fn normalise(w : u32) {
  var n = 0.0;
  for (var i = 0u; i < 16u; i = i + 1u) { n = n + W[w * 16u + i] * W[w * 16u + i]; }
  n = sqrt(max(n, 1e-12));
  for (var i = 0u; i < 16u; i = i + 1u) { W[w * 16u + i] = W[w * 16u + i] / n; }
}
// One step of gradient descent on the logistic loss of (s, r, o) with truth y.
fn learn(s : u32, r : u32, o : u32, y : f32, kind : u32) {
  let g = belief(meaning(s, r, o, kind)) - y;
  let lr = Q(4u);
  let decay = Q(5u);
  let prior = Q(7u);
  let wr = (u32(Q(0u)) + r) * 16u;
  for (var i = 0u; i < 16u; i = i + 1u) {
    W[s * 16u + i] = W[s * 16u + i] - lr * MASK_E[i] * (g * G[i] + decay * W[s * 16u + i] + prior * OFF_E[i] * W[s * 16u + i]);
    W[wr + i] = W[wr + i] - lr * MASK_R[i] * (g * G[16u + i] + decay * W[wr + i] + prior * OFF_R[i] * W[wr + i]);
    W[o * 16u + i] = W[o * 16u + i] - lr * MASK_E[i] * (g * G[32u + i] + decay * W[o * 16u + i] + prior * OFF_E[i] * W[o * 16u + i]);
  }
  normalise(s);
  normalise(o);
}
)wgsl";

constexpr const char* learning_agent = R"wgsl(
  let kind = u32(J(0u));
  let nE = u32(Q(0u));
  let nR = u32(Q(1u));
  let nF = u32(Q(2u));
  let epochs = u32(Q(3u));
  let negatives = u32(Q(6u));
  let init = Q(8u);
  // What the algebra imposes: the blades (grades 0–2: 1, N A B C e, then the ten planes) a word may use.
  for (var i = 0u; i < 16u; i = i + 1u) {
    MASK_E[i] = 1.0;
    MASK_R[i] = 1.0;
    if (kind <= 1u) {
      MASK_E[i] = select(0.0, 1.0, i == 2u || i == 3u || i == 4u);   // entities: unit vectors of V_E
      MASK_R[i] = select(0.0, 1.0, i >= 6u || (kind == 1u && i == 0u)); // relations: bivectors (+ a scalar)
    }
    // The grade-consistency prior (§11.7): entities in grade 1, relations in grade 2.
    OFF_E[i] = select(0.0, select(1.0, 0.0, i >= 1u && i <= 5u), kind == 3u);
    OFF_R[i] = select(0.0, select(1.0, 0.0, i >= 6u), kind == 3u);
  }
  for (var w = 0u; w < nE + nR; w = w + 1u) {
    for (var i = 0u; i < 16u; i = i + 1u) {
      let mask = select(MASK_R[i], MASK_E[i], w < nE);
      W[w * 16u + i] = mask * init * (2.0 * U(w * 16u + i) - 1.0);
    }
    if (w < nE) { normalise(w); }
  }
  // Reading the text: every sentence, each with corrupted ones as negatives (§11.6).
  var draw = 1000u;
  for (var ep = 0u; ep < epochs; ep = ep + 1u) {
    for (var f = 0u; f < nF; f = f + 1u) {
      let s = u32(Q(16u + 4u * f));
      let r = u32(Q(17u + 4u * f));
      let o = u32(Q(18u + 4u * f));
      learn(s, r, o, Q(19u + 4u * f), kind);
      for (var k = 0u; k < negatives; k = k + 1u) {
        let e = min(nE - 1u, u32(U(draw) * f32(nE)));
        let side = U(draw + 1u) < 0.5;
        draw = draw + 2u;
        let cs = select(s, e, side);
        let co = select(e, o, side);
        if (Q(280u + (r * nE + cs) * nE + co) > 0.5) { continue; }
        learn(cs, r, co, 0.0, kind);
      }
    }
    var right = 0.0;
    for (var f = 0u; f < nF; f = f + 1u) {
      let p = belief(meaning(u32(Q(16u + 4u * f)), u32(Q(17u + 4u * f)), u32(Q(18u + 4u * f)), kind));
      right = right + judged(p, Q(19u + 4u * f));
    }
    stat(ep, right / f32(max(nF, 1u)));
  }
  // What the learner believes about every sentence of the vocabulary, and how it fares on the tests.
  var hits = array<f32, 5>(0.0, 0.0, 0.0, 0.0, 0.0);
  var counts = array<f32, 5>(0.0, 0.0, 0.0, 0.0, 0.0);
  for (var r = 0u; r < nR; r = r + 1u) {
    for (var s = 0u; s < nE; s = s + 1u) {
      for (var o = 0u; o < nE; o = o + 1u) {
        let t = (r * nE + s) * nE + o;
        let p = belief(meaning(s, r, o, kind));
        stat(60u + t, p);
        let truth = Q(480u + t);
        if (truth < -0.5) { continue; }
        let right = judged(p, truth);
        let cat = u32(Q(680u + t));
        hits[0] = hits[0] + right;
        counts[0] = counts[0] + 1.0;
        if (cat > 0u) {
          hits[cat] = hits[cat] + right;
          counts[cat] = counts[cat] + 1.0;
        }
      }
    }
  }
  for (var c = 0u; c < 4u; c = c + 1u) { stat(41u + c, hits[c] / max(counts[c], 1.0)); }
  var right = 0.0;
  for (var f = 0u; f < nF; f = f + 1u) {
    let p = belief(meaning(u32(Q(16u + 4u * f)), u32(Q(17u + 4u * f)), u32(Q(18u + 4u * f)), kind));
    right = right + judged(p, Q(19u + 4u * f));
  }
  stat(40u, right / f32(max(nF, 1u)));
  // Where the meaning lives: the share of each grade in the entities and in each relation.
  for (var w = 0u; w < nE + nR; w = w + 1u) {
    var g0 = 0.0;
    var g1 = 0.0;
    var g2 = 0.0;
    for (var i = 0u; i < 16u; i = i + 1u) {
      let x = W[w * 16u + i] * W[w * 16u + i];
      if (i == 0u) { g0 = g0 + x; } else if (i <= 5u) { g1 = g1 + x; } else { g2 = g2 + x; }
    }
    let total = max(g0 + g1 + g2, 1e-12);
    if (w < nE) {
      stat(45u, g0 / total / f32(nE));
      stat(46u, g1 / total / f32(nE));
      stat(47u, g2 / total / f32(nE));
    } else {
      let r = w - nE;
      stat(48u + 3u * r, g0 / total);
      stat(49u + 3u * r, g1 / total);
      stat(50u + 3u * r, g2 / total);
    }
  }
)wgsl";

namespace syn::models {

namespace {

constexpr int kMaxEntities = 8, kMaxRelations = 3, kMaxSentences = 64, kMaxEpochs = 40;
const char* kKinds[5] = {"typed: entities unit vectors, relations bivectors (Layer 1)", "typed, relations with a scalar part",
                         "untyped multivectors (grades 0–2)", "untyped + grade-consistency prior", "flat: the same 16 numbers, no algebra"};

struct Sentence {
  int s, r, o;
  bool truth;
  std::string text;
};

struct Corpus {
  std::vector<std::string> entities, relations;
  std::vector<Sentence> sentences;
  std::vector<std::string> ignored;
  bool world = false;
  std::vector<int> truth, category;  // [r][s][o]: −1 unknown / 0 / 1; 0 none, 1 reversed, 2 transitive, 3 symmetric
};

int indexOf(std::vector<std::string>& v, const std::string& x, std::size_t limit) {
  const auto it = std::find(v.begin(), v.end(), x);
  if (it != v.end()) return static_cast<int>(it - v.begin());
  if (v.size() >= limit) return -1;
  v.push_back(x);
  return static_cast<int>(v.size()) - 1;
}

// "Ann is older than Ben." → (Ann, "is older than", Ben, true); a "not" makes it false.
Corpus parse(const std::string& text) {
  Corpus c;
  std::string sentence;
  auto flush = [&]() {
    std::string original = sentence;
    original.erase(0, original.find_first_not_of(" \t\r"));
    original.erase(original.find_last_not_of(" \t\r") + 1);
    std::vector<std::string> words;
    std::istringstream in(sentence);
    std::string w;
    bool negated = false;
    while (in >> w) {
      std::string clean;
      for (char ch : w)
        if (std::isalnum(static_cast<unsigned char>(ch)) || ch == '\'' || ch == '-') clean += ch;
      if (clean.empty()) continue;
      std::string lower = clean;
      std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
      if (lower == "not" || lower == "never") {
        negated = true;
        continue;
      }
      if (lower == "isn't") {
        negated = true;
        clean = "is";
      }
      words.push_back(clean);
    }
    sentence.clear();
    // Articles go ("the book" is the entity book); an auxiliary goes too, its verb taking the
    // third person ("does not love" → "loves"), so a negation names the same relation.
    std::vector<std::string> kept;
    bool third = false;
    for (std::size_t i = 0; i < words.size(); ++i) {
      std::string lower = words[i];
      std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
      if (lower == "the" || lower == "a" || lower == "an") continue;
      if ((lower == "does" || lower == "do" || lower == "did") && i > 0 && i + 2 < words.size()) {
        third = lower == "does";
        continue;
      }
      if (third) {
        if (lower.back() != 's') words[i] += "s";
        third = false;
      }
      kept.push_back(words[i]);
    }
    words = kept;
    if (words.size() < 3) return;
    std::string rel;
    for (std::size_t i = 1; i + 1 < words.size(); ++i) rel += (rel.empty() ? "" : " ") + words[i];
    const std::string shown = original + ".";
    if (static_cast<int>(c.sentences.size()) >= kMaxSentences) {
      c.ignored.push_back(shown);
      return;
    }
    std::vector<std::string> ents = c.entities, rels = c.relations;
    const int s = indexOf(ents, words.front(), kMaxEntities), o = indexOf(ents, words.back(), kMaxEntities), r = indexOf(rels, rel, kMaxRelations);
    if (s < 0 || o < 0 || r < 0) {
      c.ignored.push_back(shown);
      return;
    }
    c.entities = ents;
    c.relations = rels;
    c.sentences.push_back({s, r, o, !negated, shown});
  };
  for (char ch : text) {
    if (ch == '.' || ch == '\n' || ch == '!' || ch == ';') flush();
    else sentence += ch;
  }
  flush();
  return c;
}

// The default text: a family of six. Ages fall from Ann to Fay; Ann and Cal, Ben and Fay, Dee and Eve are siblings.
Corpus family() {
  const std::vector<std::string> names = {"Ann", "Ben", "Cal", "Dee", "Eve", "Fay"};
  std::string text;
  const int older[][2] = {{0, 1}, {1, 2}, {2, 3}, {3, 4}, {4, 5}, {0, 3}, {1, 4}, {2, 5}};
  for (const auto& p : older) text += names[p[0]] + " is older than " + names[p[1]] + ". ";
  text += "Ann is a sibling of Cal. Fay is a sibling of Ben. Dee is a sibling of Eve. Ann is not a sibling of Ben. Cal is not a sibling of Eve.";
  Corpus c = parse(text);
  c.world = true;
  const int n = static_cast<int>(c.entities.size());
  auto id = [&](const std::string& x) { return static_cast<int>(std::find(c.entities.begin(), c.entities.end(), x) - c.entities.begin()); };
  std::vector<int> age(n);
  for (int k = 0; k < 6; ++k) age[id(names[k])] = 6 - k;
  const std::array<std::array<int, 2>, 3> sib = {{{id("Ann"), id("Cal")}, {id("Ben"), id("Fay")}, {id("Dee"), id("Eve")}}};
  c.truth.assign(2 * n * n, 0);
  c.category.assign(2 * n * n, 0);
  for (int s = 0; s < n; ++s)
    for (int o = 0; o < n; ++o) {
      c.truth[(0 * n + s) * n + o] = age[s] > age[o];
      bool sibling = false;
      for (const auto& p : sib) sibling = sibling || (p[0] == s && p[1] == o) || (p[0] == o && p[1] == s);
      c.truth[(1 * n + s) * n + o] = sibling;
    }
  auto stated = [&](int s, int r, int o) {
    for (const auto& x : c.sentences)
      if (x.s == s && x.r == r && x.o == o) return true;
    return false;
  };
  for (const auto& x : c.sentences) {
    if (!x.truth) continue;
    const int rev = (x.r * n + x.o) * n + x.s;
    if (!stated(x.o, x.r, x.s)) c.category[rev] = x.r == 0 ? 1 : 3;
  }
  for (int s = 0; s < n; ++s)
    for (int o = 0; o < n; ++o)
      if (age[s] > age[o] && !stated(s, 0, o)) c.category[(0 * n + s) * n + o] = 2;
  return c;
}

}  // namespace

const kernel::Model* fgaLearningModel() {
  static const kernel::Model model = [] {
    kernel::Model m;
    m.name = "fgaLearning";
    m.title = "Learning meaning in the space: learners read a text and learn multivectors of Cl(4, 1) (functional geometric algebra)";
    m.maxStats = 252;
    m.agentSource = learning_agent;
    m.helpersWgsl = space::algebraWgsl(1) + learning_helpers;
    m.compile = [](const Value& spec) {
      const Value& c = spec.sub("fgaLearning");
      const std::string text = c.str("text", "");
      Corpus corpus = text.empty() ? family() : parse(text);
      const int nE = static_cast<int>(corpus.entities.size()), nR = static_cast<int>(corpus.relations.size());
      std::vector<double> q = {double(nE), double(nR), double(corpus.sentences.size()), double(std::clamp(c.integer("epochs", 30), 1, kMaxEpochs)),
                               c.num("rate", 0.3), c.num("decay", 0.002), double(std::clamp(c.integer("negatives", 1), 0, 6)), c.num("prior", 0.3),
                               c.num("init", 0.5)};
      q.resize(16, 0.0);
      for (const auto& s : corpus.sentences) q.insert(q.end(), {double(s.s), double(s.r), double(s.o), s.truth ? 1.0 : 0.0});
      q.resize(280, 0.0);
      const int cells = nR * nE * nE;
      std::vector<double> stated(cells, 0.0), truth(cells, -1.0), category(cells, 0.0);
      for (const auto& s : corpus.sentences)
        if (s.truth) stated[(s.r * nE + s.s) * nE + s.o] = 1.0;
      for (int t = 0; t < cells; ++t) {
        if (corpus.world) {
          truth[t] = corpus.truth[t];
          category[t] = corpus.category[t];
        }
      }
      if (!corpus.world)
        for (const auto& s : corpus.sentences) truth[(s.r * nE + s.s) * nE + s.o] = s.truth ? 1.0 : 0.0;
      q.insert(q.end(), stated.begin(), stated.end());
      q.resize(480, 0.0);
      q.insert(q.end(), truth.begin(), truth.end());
      q.resize(680, 0.0);
      q.insert(q.end(), category.begin(), category.end());
      q.resize(880, 0.0);
      // The terms of ⟨S R O⟩₀ over the blades of grades 0–2, for the blades each kind of learner uses.
      const std::vector<unsigned> blades = {0, 1, 2, 4, 8, 16, 3, 5, 6, 9, 10, 12, 17, 18, 20, 24};
      auto entityBlade = [](int list, unsigned i) { return list == 2 || i == 2 || i == 3 || i == 4; };
      auto relationBlade = [](int list, unsigned i) { return list == 2 || i >= 6 || (list == 1 && i == 0); };
      int term = 0;
      for (int list = 0; list < 3; ++list) {
        q[10 + list] = term;
        int count = 0;
        for (unsigned a = 0; a < 16; ++a)
          for (unsigned b = 0; b < 16; ++b) {
            const unsigned cm = blades[a] ^ blades[b];
            const auto it = std::find(blades.begin(), blades.end(), cm);
            if (it == blades.end()) continue;
            const auto ci = static_cast<unsigned>(it - blades.begin());
            if (!entityBlade(list, a) || !relationBlade(list, b) || !entityBlade(list, ci)) continue;
            const double tau = Multivector::bladeSign(blades[a], blades[b], 4) * Multivector::bladeSign(cm, cm, 4);
            q.insert(q.end(), {double(a), double(b), double(ci), tau});
            ++count;
            ++term;
          }
        q[13 + list] = count;
      }
      Value meta = Value::object();
      meta.set("entities", corpus.entities).set("relations", corpus.relations).set("world", corpus.world).set("ignored", corpus.ignored);
      Value sentences = Value::array();
      for (const auto& s : corpus.sentences) sentences.push(Value::object().set("text", s.text).set("truth", s.truth));
      meta.set("sentences", sentences);
      Value cats = Value::array();
      for (int t = 0; t < cells; ++t) cats.push(Value(static_cast<double>(category[t])));
      meta.set("category", cats);
      std::vector<std::vector<double>> rows;
      for (int k = 0; k < 5; ++k) rows.push_back({double(k)});
      Plan plan = makePlan("fgaLearning", static_cast<std::uint32_t>(c.integer("learners", 64)), 252, space::withSignTable(q), rows, seedOf(spec));
      plan.meta = meta;
      return std::vector<Plan>{plan};
    };
    m.summarise = [](const Value&, const std::vector<Plan>& plans, const std::vector<Output>& out) {
      const Plan& p = plans[0];
      const Output& o = out[0];
      const Value& meta = p.meta;
      const auto& entities = meta.at("entities").asArray();
      const auto& relations = meta.at("relations").asArray();
      const int nE = static_cast<int>(entities.size()), nR = static_cast<int>(relations.size());
      const int epochs = static_cast<int>(p.params[space::kSignTable + 3]);
      Value designs = Value::array();
      for (std::uint32_t j = 0; j < p.jobs; ++j) {
        Value rel = Value::array(), truth = Value::array();
        for (int r = 0; r < nR; ++r) {
          rel.push(Value::Numbers(o.range(p, j, 48 + 3 * r, 3)));
          Value rows = Value::array();
          for (int s = 0; s < nE; ++s) rows.push(Value::Numbers(o.range(p, j, 60 + (r * nE + s) * nE, nE)));
          truth.push(rows);
        }
        designs.push(Value::object()
                         .set("name", kKinds[j])
                         .set("curve", Value::Numbers(o.range(p, j, 0, epochs)))
                         .set("read", o.mean(p, j, 40))
                         .set("world", o.mean(p, j, 41))
                         .set("reversed", o.mean(p, j, 42))
                         .set("transitive", o.mean(p, j, 43))
                         .set("symmetric", o.mean(p, j, 44))
                         .set("entityGrades", Value::Numbers(o.range(p, j, 45, 3)))
                         .set("relationGrades", rel)
                         .set("belief", truth));
      }
      int counts[4] = {0, 0, 0, 0};
      for (const auto& c : meta.at("category").asArray()) counts[static_cast<int>(c.asNumber())]++;
      return Value::object()
          .set("designs", designs)
          .set("entities", meta.at("entities"))
          .set("relations", meta.at("relations"))
          .set("sentences", meta.at("sentences"))
          .set("ignored", meta.at("ignored"))
          .set("world", meta.at("world"))
          .set("tests", Value::object().set("reversed", double(counts[1])).set("transitive", double(counts[2])).set("symmetric", double(counts[3])))
          .set("learners", static_cast<double>(p.agents));
    };
    return m;
  }();
  return &model;
}

}  // namespace syn::models
