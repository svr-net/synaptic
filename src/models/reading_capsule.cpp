// The reader's knowledge capsule: what it reads, ingested into the space through an index router of
// grades, packed smaller than the text, and recalled from it.
//
// Ingesting (capsuleIngest): the grammar reads every clause of a passage as a fact — subject,
// relation (the verb and what goes with it), object — in the space. What it knows of each word on
// its own is a handful of cues (its word class's closed lists, its ending, its capital, its
// neighbour); the cues dilate a point of the space (they add on its logit), and the word is the
// verb if its point lies inside the verb class's cap — a sphere of the space, tested by the inner
// product. The verb is the clause's first word inside the cap; the words that go with it lie inside
// a second cap; the words before it are the subject, the rest the object. Every word of a fact then
// becomes one multivector of Cl(4, 1), a slot, whose grades are the levels of the index:
//   grade 0 (scalar)    the fact it belongs to;
//   grade 1 (vector)    the word: the point of S³ its coordinates lift to;
//   grade 2 (bivector)  the grammar: θ/2 · e_N∧e_A for its role, φ/2 · e_B∧e_C for its place —
//                       the generator of the rotor that binds the word to its role and place.
// The index router isolates each grade (⟨M⟩_k) and routes it to its level's codebook: grade 1 to
// the nearest point of the passage's vocabulary, grade 2 to the grammar's lattice of 3 roles × 32
// places, grade 0 to the run of its fact. The capsule keeps only those indices — per passage its
// vocabulary (indices into a lexicon of the words it knows), its facts as run lengths, and a code
// per slot, packed in as many bits as the codebooks need; the text is not kept. Agents are the
// words of a passage's clauses, jobs are passages.
//
// Recalling (capsuleRecall): from the codes alone, each slot's multivector is rebuilt — its word's
// point (grade 1) and the grammar's bivector (grade 2) — the rotor exp(−B) is formed from the
// isolated grade 2, the word is bound by it (R w R̃), and the role and place are read back off the
// bound point's rotation angles (grammarOf) and checked against the code. The facts are put back in
// order by the places alone (each fact's slots are kept out of order). readingQA and readingStudent
// construct answers from the recalled facts (reading_memory.hpp), never from the text.
#include <algorithm>
#include <array>
#include <cctype>
#include <map>
#include <set>
#include <string>

#include "reading_memory.hpp"
#include "reading_text.hpp"
#include "syn/models/models.hpp"
#include "syn/space/algebra.hpp"

// The grammar's word classes, in WGSL: a word's cues dilate the equator of the cognitive channel
// (each adds its weight to the logit), and a class is a cap of the space.
constexpr const char* word_classes = R"wgsl(
// Q(3) auxiliary, Q(4) verb ending, Q(5) each sign against (function word, capital, after a
// preposition, first word), Q(6) goes with a verb, Q(7) after a plural first word, Q(8) the
// chance the verb class's cap holds.
fn cueOf(t : u32) -> u32 { return u32(Q(u32(Q(1u)) + 5u * t + 1u)); }
fn against(c : u32) -> f32 {
  var n = 0.0;
  for (var b = 2u; b <= 5u; b = b + 1u) { n = n + select(0.0, 1.0, ((c >> b) & 1u) != 0u); }
  return n * Q(5u);
}
// The verb class: the point of the word's cues inside the cap.
fn isVerb(t : u32) -> bool {
  let c = cueOf(t);
  let x = select(0.0, Q(3u), (c & 1u) != 0u) + select(0.0, Q(4u), (c & 2u) != 0u) - against(c);
  return inside(atLogit(x, kA), chanceCap(Q(8u)));
}
// After a plural first word, a plain word is the verb ("Honeybees communicate"), when no word is.
fn isVerbAfterPlural(t : u32) -> bool {
  let c = cueOf(t);
  let x = select(0.0, Q(7u), (c & 128u) != 0u) - Q(5u) * (select(0.0, 1.0, (c & 4u) != 0u) + select(0.0, 1.0, (c & 8u) != 0u));
  return inside(atLogit(x, kA), chanceCap(Q(8u)));
}
// The class of the words that go with a verb ("is used to", "can also help").
fn goesWithVerb(t : u32) -> bool {
  let x = select(-Q(6u), Q(6u), (cueOf(t) & 64u) != 0u);
  return inside(atLogit(x, kA), chanceCap(Q(8u)));
}
)wgsl";

// Ingesting, in WGSL: each word's role found in the space, the slot as a multivector, routed grade
// by grade to its code.
constexpr const char* ingest_agent = R"wgsl(
  // Constants (after the sign table): Q(0) start of the lexicon's coordinates (3 per word), Q(1)
  // start of the words (lexicon index, cues, first word of its clause, clause length, fact), Q(2)
  // start of the vocabularies, Q(3..8) the word classes. Job: J(0) first word of the passage,
  // J(1) its words, J(2) first word of its vocabulary, J(3) its words.
  if (A >= u32(J(1u))) {
    quant(0u, -1.0);
    return;
  }
  let t = u32(J(0u)) + A;
  let b = u32(Q(1u)) + 5u * t;
  let first = u32(Q(b + 2u));
  let n = u32(Q(b + 3u));
  let i = t - first;
  // The verb: the clause's first word inside the verb class's cap.
  var verb = n;
  for (var k = 0u; k < n; k = k + 1u) {
    if (verb == n && isVerb(first + k)) { verb = k; }
  }
  if (verb == n && n > 2u && isVerbAfterPlural(first + 1u)) { verb = 1u; }
  // The word's role and place in the fact; words beyond the fact's reach are not kept.
  var role = 2u;
  var place = i;
  var kept = i < 24u;
  if (verb < n) {
    var end = verb + 1u;
    for (var k = verb + 1u; k < min(n, verb + 4u); k = k + 1u) {
      if (end == k && goesWithVerb(first + k)) { end = k + 1u; }
    }
    let start = select(0u, verb - 8u, verb > 8u);
    place = i - min(i, start);
    if (i < verb) {
      role = 0u;
      kept = i >= start;
    } else if (i < end) {
      role = 1u;
      kept = true;
    } else {
      kept = i < end + 16u;
    }
  }
  if (!kept) {
    quant(0u, -1.0);
    return;
  }
  let u = u32(Q(0u)) + 3u * u32(Q(b));
  // The slot: grade 0 its fact, grade 1 its word's point, grade 2 the grammar's bivector.
  mvVector(0u, lift(Q(u), Q(u + 1u), Q(u + 2u), 1.0));
  mvSet(0u, 0u, Q(b + 4u));
  mvSet(0u, 3u, 0.5 * (0.5 + kGrammarRole * f32(role)));
  mvSet(0u, 12u, 0.5 * (0.3 + kGrammarStep * f32(place)));
  // The index router. Level 1: grade 1 isolated, routed to the nearest point of the vocabulary.
  mvGrade(1u, 0u, 1u);
  let x = V5(vec4<f32>(mvGet(1u, 1u), mvGet(1u, 2u), mvGet(1u, 4u), mvGet(1u, 8u)), mvGet(1u, 16u));
  var word = 0u;
  var nearest = -1e30;
  for (var k = 0u; k < u32(J(3u)); k = k + 1u) {
    let v = u32(Q(0u)) + 3u * u32(Q(u32(Q(2u)) + u32(J(2u)) + k));
    let d = dot5(x, lift(Q(v), Q(v + 1u), Q(v + 2u), 1.0));
    if (d > nearest) { nearest = d; word = k; }
  }
  // Level 2: grade 2 isolated, routed to the grammar's lattice of roles and places.
  mvGrade(2u, 0u, 2u);
  let r = u32(clamp(round((2.0 * mvGet(2u, 3u) - 0.5) / kGrammarRole), 0.0, 2.0));
  let p = u32(clamp(round((2.0 * mvGet(2u, 12u) - 0.3) / kGrammarStep), 0.0, 31.0));
  // Level 0: grade 0 isolated — the fact, which the capsule keeps as runs of slots.
  mvGrade(3u, 0u, 0u);
  stat(0u, select(0.0, 1.0, abs(mvGet(3u, 0u) - Q(b + 4u)) < 0.5 && nearest > -1e-5));
  quant(0u, f32((word * 3u + r) * 32u + p));
)wgsl";

// Recalling, in WGSL: the multivector rebuilt from the code, the rotor from its grade 2.
constexpr const char* recall_agent = R"wgsl(
  // Constants (after the sign table): Q(0) start of the lexicon's coordinates, Q(1) start of the
  // vocabularies, Q(2) start of the codes. Job: J(0) first slot, J(1) slots, J(2) first word of
  // the passage's vocabulary, J(3) its words.
  if (A >= u32(J(1u))) {
    quant(0u, -1.0);
    return;
  }
  let code = u32(Q(u32(Q(2u)) + u32(J(0u)) + A));
  let k = code / 96u;
  if (k >= u32(J(3u))) {
    quant(0u, -1.0);
    return;
  }
  let word = u32(Q(u32(Q(1u)) + u32(J(2u)) + k));
  let u = u32(Q(0u)) + 3u * word;
  let w = lift(Q(u), Q(u + 1u), Q(u + 2u), 1.0);
  // The slot rebuilt: grade 1 the word's point, grade 2 the grammar's bivector B.
  mvVector(0u, w);
  mvSet(0u, 3u, 0.5 * (0.5 + kGrammarRole * f32((code / 32u) % 3u)));
  mvSet(0u, 12u, 0.5 * (0.3 + kGrammarStep * f32(code % 32u)));
  // The rotor exp(−B) from the isolated grade 2: the product of its two plane rotors.
  mvGrade(1u, 0u, 2u);
  mvPlaneRotor(2u, 0u, 1u, 2.0 * mvGet(1u, 3u));
  mvPlaneRotor(3u, 2u, 3u, 2.0 * mvGet(1u, 12u));
  mvMul(4u, 2u, 3u, 0u);
  // The word (grade 1) bound by the rotor, R w R̃; the role and place read back off the bound point.
  mvGrade(1u, 0u, 1u);
  mvSandwich(5u, 4u, 1u, 2u, 3u);
  let x = V5(vec4<f32>(mvGet(5u, 1u), mvGet(5u, 2u), mvGet(5u, 4u), mvGet(5u, 8u)), mvGet(5u, 16u));
  let g = grammarOf(x, w);
  var found = -1.0;
  if (g == i32(code % 96u)) { found = f32(u32(g) * 65536u + word); }
  quant(0u, found);
  stat(0u, select(0.0, 1.0, found >= 0.0));
)wgsl";

namespace syn::models {

namespace {

std::vector<double> coordinates(const Value& dictionary) {
  std::vector<double> out;
  for (std::size_t w = 0; w < dictionary.size(); ++w) {
    float u[3];
    memory::wordCoordinates(dictionary[w].asString(), u);
    out.insert(out.end(), {u[0], u[1], u[2]});
  }
  return out;
}

std::string helpers() { return space::algebraWgsl(6) + memory::grammarWgsl(); }

}  // namespace

const kernel::Model* capsuleIngestModel() {
  static const kernel::Model model = [] {
    kernel::Model m;
    m.name = "capsuleIngest";
    m.title = "The knowledge capsule: passages read by the grammar, every word a multivector routed grade by grade to a packed code";
    m.maxStats = 1;
    m.agentSource = ingest_agent;
    m.helpersWgsl = helpers() + word_classes;
    m.compile = [](const Value& spec) {
      const Value& c = spec.sub("capsuleIngest");
      std::vector<std::string> passages;
      if (c.has("passages"))
        for (std::size_t i = 0; i < c.at("passages").size(); ++i) passages.push_back(c.at("passages")[i].asString());
      else {
        const Value g = text::goldenSet(c.integer("questions", 60));
        for (std::size_t i = 0; i < g.at("passages").size(); ++i) passages.push_back(g.at("passages")[i].asString());
      }
      if (passages.empty()) throw std::invalid_argument("capsuleIngest: no passages");
      // The clauses of every passage, with the cues of their words, and the lexicon of the words.
      std::vector<std::vector<memory::Clause>> clauses;
      std::set<std::string> words;
      double read = 0, textBytes = 0;
      for (const auto& p : passages) {
        clauses.push_back(memory::clauses(p));
        read += static_cast<double>(text::words(p, false).size());
        textBytes += static_cast<double>(p.size());
        for (const auto& cl : clauses.back()) words.insert(cl.words.begin(), cl.words.end());
      }
      Value dictionary = Value::array();
      std::map<std::string, int> index;
      for (const auto& w : words) {
        index[w] = static_cast<int>(dictionary.size());
        dictionary.push(w);
      }
      // The word classes: auxiliary, verb ending, each sign against, goes with a verb, after a plural
      // first word, and the chance the verb class's cap holds.
      std::vector<double> q = {0, 0, 0, c.num("auxiliary", 20), c.num("verbEnding", 2), c.num("against", 4), c.num("goesWithVerb", 2),
                               c.num("afterPlural", 2), c.num("verbCap", 0.75)};
      q[0] = double(q.size());
      const auto coords = coordinates(dictionary);
      q.insert(q.end(), coords.begin(), coords.end());
      q[1] = double(q.size());
      std::vector<std::set<double>> vocabularies(clauses.size());
      std::vector<std::vector<double>> rows;
      Value shape = Value::array();
      std::size_t token = 0, most = 1;
      std::vector<double> firsts;
      std::vector<Value> clauseLengths;
      for (std::size_t p = 0; p < clauses.size(); ++p) {
        const std::size_t start = token;
        Value lengths = Value::array();
        for (std::size_t fi = 0; fi < clauses[p].size(); ++fi) {
          const auto& cl = clauses[p][fi];
          const std::size_t clauseFirst = token;
          for (std::size_t k = 0; k < cl.words.size(); ++k, ++token) {
            q.insert(q.end(), {double(index[cl.words[k]]), double(cl.cues[k]), double(clauseFirst), double(cl.words.size()), double(fi)});
            vocabularies[p].insert(index[cl.words[k]]);
          }
          lengths.push(static_cast<double>(cl.words.size()));
        }
        firsts.push_back(double(start));
        clauseLengths.push_back(lengths);
        most = std::max(most, token - start);
      }
      q[2] = double(q.size());
      for (std::size_t p = 0; p < clauses.size(); ++p) {
        const double vocabStart = double(q.size()) - q[2];
        q.insert(q.end(), vocabularies[p].begin(), vocabularies[p].end());
        const double count = p + 1 < clauses.size() ? firsts[p + 1] - firsts[p] : double(token) - firsts[p];
        rows.push_back({firsts[p], count, vocabStart, double(vocabularies[p].size())});
        shape.push(Value::object().set("clauses", clauseLengths[p]).set("vocabulary", Value::Numbers(std::vector<double>(vocabularies[p].begin(), vocabularies[p].end()))));
      }
      Plan p = makePlan("capsuleIngest", static_cast<std::uint32_t>(most), 1, space::withSignTable(q), rows, seedOf(spec), 1);
      p.keepAgents = p.agents;
      p.meta = Value::object().set("dictionary", dictionary).set("shape", shape).set("read", read).set("textBytes", textBytes);
      return std::vector<Plan>{p};
    };
    m.summarise = [](const Value&, const std::vector<Plan>& plans, const std::vector<Output>& out) {
      const Plan& p = plans[0];
      const auto& v = out[0].agentValues;
      Value passages = Value::array();
      double slots = 0, facts = 0;
      for (std::uint32_t j = 0; j < p.jobs; ++j) {
        const Value& shape = p.meta.at("shape").asArray()[j];
        const Value& routing = shape.at("vocabulary");
        // The kept words, fact by fact, and the vocabulary they use (level 1's codebook, re-indexed).
        std::vector<std::vector<std::uint32_t>> kept;
        std::set<double> used;
        std::size_t a = 0;
        const Value& lengths = shape.at("clauses");
        for (std::size_t f = 0; f < lengths.size(); ++f) {
          std::vector<std::uint32_t> fact;
          for (std::size_t k = 0; k < static_cast<std::size_t>(lengths[f].asNumber()); ++k, ++a) {
            const double code = v[static_cast<std::size_t>(j) * p.keepAgents + a];
            if (code < 0) continue;
            fact.push_back(static_cast<std::uint32_t>(code));
            used.insert(routing[static_cast<std::uint32_t>(code) / 96u].asNumber());
          }
          kept.push_back(fact);
        }
        const std::vector<double> vocabulary(used.begin(), used.end());
        std::vector<double> sizes, codes;
        for (const auto& fact : kept) {
          // Kept out of order (odd places, then even ones backwards): the order is in the grammar's bivectors.
          std::vector<std::size_t> order;
          for (std::size_t k = 1; k < fact.size(); k += 2) order.push_back(k);
          for (std::size_t k = fact.size(); k-- > 0;)
            if (k % 2 == 0) order.push_back(k);
          for (std::size_t k : order) {
            const std::uint32_t c = fact[k];
            const double w = routing[c / 96u].asNumber();
            const auto at = static_cast<std::uint32_t>(std::lower_bound(vocabulary.begin(), vocabulary.end(), w) - vocabulary.begin());
            codes.push_back(double(at * 96u + c % 96u));
          }
          sizes.push_back(double(fact.size()));
        }
        passages.push(Value::object().set("vocabulary", Value::Numbers(vocabulary)).set("facts", Value::Numbers(sizes)).set("codes", Value::Numbers(codes)));
        slots += static_cast<double>(codes.size());
        facts += static_cast<double>(sizes.size());
      }
      // The lexicon: only the words the capsule keeps (the grammar drops words beyond a fact's reach).
      std::set<double> known;
      for (const auto& ps : passages.asArray())
        for (const auto& w : ps.at("vocabulary").asNumbers()) known.insert(w);
      const std::vector<double> keep(known.begin(), known.end());
      Value dictionary = Value::array();
      for (double w : keep) dictionary.push(p.meta.at("dictionary")[static_cast<std::size_t>(w)]);
      Value relexed = Value::array();
      for (const auto& ps : passages.asArray()) {
        std::vector<double> vocabulary;
        for (double w : ps.at("vocabulary").asNumbers()) vocabulary.push_back(double(std::lower_bound(keep.begin(), keep.end(), w) - keep.begin()));
        relexed.push(Value::object().set("vocabulary", Value::Numbers(vocabulary)).set("facts", ps.at("facts")).set("codes", ps.at("codes")));
      }
      const Value capsule = Value::object().set("dictionary", dictionary).set("passages", relexed);
      const memory::CapsuleBytes packed = memory::packCapsule(capsule);
      const double text = p.meta.num("textBytes", 0);
      return Value::object()
          .set("capsule", capsule)
          .set("packed", packed.base64)
          .set("bytes", Value::object()
                            .set("lexicon", static_cast<double>(packed.lexicon))
                            .set("vocabulary", static_cast<double>(packed.vocabulary))
                            .set("facts", static_cast<double>(packed.facts))
                            .set("codes", static_cast<double>(packed.codes))
                            .set("total", static_cast<double>(packed.bytes.size()))
                            .set("text", text)
                            .set("ratio", text > 0 ? static_cast<double>(packed.bytes.size()) / text : 0.0))
          .set("routed", [&] {
            // The share of slots whose grades routed back to their word and fact exactly.
            double sum = 0;
            for (std::uint32_t j = 0; j < p.jobs; ++j) sum += out[0].mean(p, j, 0) * p.agents;
            return slots > 0 ? sum / slots : 0.0;
          }())
          .set("passages", static_cast<double>(p.jobs))
          .set("facts", facts)
          .set("points", slots)
          .set("lexicon", static_cast<double>(dictionary.size()))
          .set("wordsRead", p.meta.at("read"));
    };
    return m;
  }();
  return &model;
}

const kernel::Model* capsuleRecallModel() {
  static const kernel::Model model = [] {
    kernel::Model m;
    m.name = "capsuleRecall";
    m.title = "Recalling the knowledge capsule: each code rebuilt as a multivector, its grade 2 the rotor that gives back the word's role and place";
    m.maxStats = 1;
    m.agentSource = recall_agent;
    m.helpersWgsl = helpers();
    m.compile = [](const Value& spec) {
      const Value& c = spec.sub("capsuleRecall");
      const Value unpacked = c.has("packed") ? memory::unpackCapsule(c.at("packed").asString()) : Value();
      const Value& capsule = c.has("packed") ? unpacked : c.has("capsule") ? c.at("capsule") : memory::storedCapsule();
      const Value& dictionary = capsule.at("dictionary");
      const auto& passages = capsule.at("passages").asArray();
      if (passages.empty()) throw std::invalid_argument("capsuleRecall: an empty capsule");
      std::vector<double> q = {0, 0, 0};
      q[0] = double(q.size());
      const auto coords = coordinates(dictionary);
      q.insert(q.end(), coords.begin(), coords.end());
      q[1] = double(q.size());
      std::vector<double> starts;
      for (const auto& p : passages) {
        starts.push_back(double(q.size()) - q[1]);
        const Value& vocab = p.at("vocabulary");
        for (std::size_t k = 0; k < vocab.size(); ++k) q.push_back(vocab[k].asNumber());
      }
      q[2] = double(q.size());
      std::vector<std::vector<double>> rows;
      std::size_t first = 0, most = 1;
      Value shape = Value::array();
      for (std::size_t i = 0; i < passages.size(); ++i) {
        const Value& codes = passages[i].at("codes");
        for (std::size_t k = 0; k < codes.size(); ++k) q.push_back(codes[k].asNumber());
        rows.push_back({double(first), double(codes.size()), starts[i], double(passages[i].at("vocabulary").size())});
        shape.push(passages[i].at("facts"));
        first += codes.size();
        most = std::max(most, codes.size());
      }
      Plan p = makePlan("capsuleRecall", static_cast<std::uint32_t>(most), 1, space::withSignTable(q), rows, seedOf(spec), 1);
      p.keepAgents = p.agents;
      p.meta = Value::object().set("dictionary", dictionary).set("shape", shape);
      return std::vector<Plan>{p};
    };
    m.summarise = [](const Value&, const std::vector<Plan>& plans, const std::vector<Output>& out) {
      const Plan& p = plans[0];
      const auto& v = out[0].agentValues;
      Value passages = Value::array();
      double slots = 0, recovered = 0;
      for (std::uint32_t j = 0; j < p.jobs; ++j) {
        const Value& sizes = p.meta.at("shape").asArray()[j];
        Value facts = Value::array();
        std::size_t a = 0;
        for (std::size_t f = 0; f < sizes.size(); ++f) {
          // Each recovered word with the grammar's rotor that placed it: put back in order by the places.
          std::vector<std::array<std::uint32_t, 3>> heard;  // (place, role, word)
          for (std::size_t k = 0; k < static_cast<std::size_t>(sizes[f].asNumber()); ++k, ++a) {
            const double code = v[static_cast<std::size_t>(j) * p.keepAgents + a];
            slots += 1;
            if (code < 0) continue;  // not recovered: the rotor did not give its role and place back
            recovered += 1;
            const auto c = static_cast<std::uint32_t>(code);
            heard.push_back({(c / 65536u) % 32u, std::min<std::uint32_t>(c / 65536u / 32u, 2u), c % 65536u});
          }
          std::sort(heard.begin(), heard.end());
          std::vector<double> parts[3];
          for (const auto& h : heard) parts[h[1]].push_back(double(h[2]));
          Value fact = Value::object();
          for (int r = 0; r < memory::kRoles; ++r) fact.set(memory::kRoleNames[r], Value::Numbers(parts[r]));
          facts.push(fact);
        }
        passages.push(Value::object().set("facts", facts));
      }
      return Value::object()
          .set("recalled", Value::object().set("dictionary", p.meta.at("dictionary")).set("passages", passages))
          .set("points", slots)
          .set("recovered", slots > 0 ? recovered / slots : 0.0);
    };
    return m;
  }();
  return &model;
}

}  // namespace syn::models

// ---- capsuleQA: recall, reasoning and answering, in one kernel, from the capsule alone.

// Reasoning and answering, in WGSL: one agent per way of saying one fact for one part of a question.
constexpr const char* qa_agent = R"wgsl(
  // Constants: P(0) start of the lexicon's coordinates (3 per word), P(1) start of the codes, P(2)
  // start of the vocabularies, P(3) start of the asked words (idf, first and count of their lexicon
  // forms), P(4) start of the lexicon forms, P(5..12) the constructing habits, P(13) τ, P(14) start
  // of the facts (first slot, slots, first word of its capsule's vocabulary), P(15) start of the
  // lexicon's flags (1 a digit, 2 a capital). Job (a part of a question): J(0) its first fact,
  // J(1) facts, J(3) first asked word, J(4) asked words, J(5) asks a number, J(6) asks a name,
  // J(7) the question's parts.
  let nf = u32(J(1u));
  let f = A / 5u;
  let pattern = A % 5u;
  // One sentence per part: a question in parts says one fact for each (Claude's habit 4).
  if (f >= nf || (pattern == 4u && (f + 1u >= nf || J(7u) > 1.5))) {
    for (var k = 0u; k < 9u; k = k + 1u) { quant(k, 0.0); }
    return;
  }
  let asked = min(u32(J(4u)), 16u);
  var held : array<bool, 16>;
  var said : array<bool, 16>;
  for (var t = 0u; t < 16u; t = t + 1u) {
    held[t] = false;
    said[t] = false;
  }
  var count = 0u;
  var fit = false;
  var firstPlace = 99u;
  for (var stage = 0u; stage < 2u; stage = stage + 1u) {
    for (var which = 0u; which < select(1u, 2u, pattern == 4u); which = which + 1u) {
      let fact = u32(P(14u)) + 3u * (u32(J(0u)) + f + which);
      for (var s = 0u; s < u32(P(fact + 1u)); s = s + 1u) {
        let code = u32(P(u32(P(1u)) + u32(P(fact)) + s));
        let role = (code / 32u) % 3u;
        let place = code % 32u;
        let word = u32(P(u32(P(2u)) + u32(P(fact + 2u)) + code / 96u));
        let c = u32(P(0u)) + 3u * word;
        let w = lift(P(c), P(c + 1u), P(c + 2u), 1.0);
        // Recall: the grammar's rotor of the code turns the word, and is read back off the point.
        if (grammarOf(bindWord(w, role, place), w) != i32(code % 96u)) { continue; }
        let inSaid = pattern == 0u || pattern == 4u || (pattern == 1u && role >= 1u) || (pattern == 2u && role == 2u) || (pattern == 3u && role == 0u);
        if (stage == 0u) {
          if (inSaid && which == 0u) { firstPlace = min(firstPlace, place); }
          continue;
        }
        // Evidence: the question's words in the fact, by identity of their points.
        for (var t = 0u; t < asked; t = t + 1u) {
          let row = u32(P(3u)) + 3u * (u32(J(3u)) + t);
          for (var l = 0u; l < u32(P(row + 2u)); l = l + 1u) {
            let o = u32(P(0u)) + 3u * u32(P(u32(P(4u)) + u32(P(row + 1u)) + l));
            if (dot5(w, lift(P(o), P(o + 1u), P(o + 2u), 1.0)) > -1e-5) {
              if (which == 0u) { held[t] = true; }
              if (inSaid) { said[t] = true; }
            }
          }
        }
        if (inSaid) {
          count = count + 1u;
          let flags = u32(P(u32(P(15u)) + word));
          let first = which == 0u && place == firstPlace;
          fit = fit || (J(5u) > 0.5 && (flags & 1u) != 0u) || (J(6u) > 0.5 && (flags & 2u) != 0u && !first);
        }
      }
    }
  }
  if (count == 0u) {
    for (var k = 0u; k < 9u; k = k + 1u) { quant(k, 0.0); }
    return;
  }
  var mass = 0.0;
  var weight = 0.0;
  var covered = 0.0;
  for (var t = 0u; t < asked; t = t + 1u) {
    let idf = P(u32(P(3u)) + 3u * (u32(J(3u)) + t));
    mass = mass + idf;
    weight = weight + select(0.0, idf, held[t]);
    covered = covered + select(0.0, 1.0, said[t]);
  }
  // The reasoning: the evidence and the ways of saying it, weighed by the constructing habits.
  var feature : array<f32, 8>;
  feature[0] = select(0.0, weight / mass, mass > 0.0);
  feature[1] = select(0.0, 1.0, pattern == 4u);
  feature[2] = select(0.0, 1.0, pattern == 0u || pattern == 4u);
  feature[3] = select(0.0, 1.0, pattern == 1u);
  feature[4] = select(0.0, 1.0, pattern == 2u);
  feature[5] = select(0.0, 1.0, pattern == 3u);
  feature[6] = select(0.0, 1.0, fit);
  feature[7] = select(0.0, covered / f32(asked), asked > 0u);
  var score = 0.0;
  for (var k = 0u; k < 8u; k = k + 1u) {
    score = score + P(5u + k) * feature[k];
    quant(1u + k, feature[k]);
  }
  // The belief that this is the answer: a point of the space.
  quant(0u, chance(atLogit(clamp(score - P(13u), -30.0, 30.0), kA)));
)wgsl";

namespace syn::models {

const kernel::Model* capsuleQaModel() {
  static const kernel::Model model = [] {
    kernel::Model m;
    m.name = "capsuleQA";
    m.title = "Reasoning and answering from the knowledge capsule alone: question parts, evidence in the space, answers constructed by the grammar";
    m.maxStats = 1;
    m.agentSource = qa_agent;
    m.helpersWgsl = memory::grammarWgsl();
    m.compile = [](const Value& spec) {
      const Value& c = spec.sub("capsuleQA");
      const Value unpacked = c.has("packed") ? memory::unpackCapsule(c.at("packed").asString()) : Value();
      const Value& capsule = c.has("packed") ? unpacked : c.has("capsule") ? c.at("capsule") : memory::storedCapsule();
      const Value& dictionary = capsule.at("dictionary");
      const auto& passages = capsule.at("passages").asArray();
      // The questions: given, or the golden set (question i about capsule i).
      std::vector<std::array<std::string, 2>> questions;
      std::vector<double> capsuleOf;
      const bool given = c.has("questions") && c.at("questions").size() > 0;  // a list, or the golden set's size
      if (given)
        for (std::size_t i = 0; i < c.at("questions").size(); ++i) {
          const Value& q = c.at("questions").asArray()[i];
          questions.push_back({q.at("question").asString(), q.str("golden", "")});
          capsuleOf.push_back(q.num("capsule", -1));
        }
      else {
        const Value g = text::goldenSet(c.integer("questions", 60));
        for (const auto& q : g.at("questions").asArray()) {
          questions.push_back({q.at("question").asString(), q.at("golden").asString()});
          capsuleOf.push_back(q.at("passage").asNumber());
        }
      }
      // The facts of every capsule: their slots (from the codes) and the stems of their words.
      std::vector<std::vector<std::array<std::size_t, 2>>> facts(passages.size());  // (first slot, slots)
      std::vector<double> codes;
      for (std::size_t p = 0; p < passages.size(); ++p) {
        const Value& sizes = passages[p].at("facts");
        const Value& cs = passages[p].at("codes");
        std::size_t at = codes.size();
        for (std::size_t k = 0; k < cs.size(); ++k) codes.push_back(cs[k].asNumber());
        for (std::size_t f = 0; f < sizes.size(); ++f) {
          const auto n = static_cast<std::size_t>(sizes[f].asNumber());
          facts[p].push_back({at, n});
          at += n;
        }
      }
      auto wordOf = [&](std::size_t p, double code) {
        return static_cast<std::size_t>(passages[p].at("vocabulary")[static_cast<std::size_t>(code) / 96].asNumber());
      };
      std::map<std::string, std::vector<double>> formsOf;  // a stem's lexicon forms
      std::vector<double> flags;
      for (std::size_t w = 0; w < dictionary.size(); ++w) {
        const std::string word = dictionary[w].asString();
        for (const auto& st : memory::stemsOf(word)) formsOf[st].push_back(double(w));
        flags.push_back((std::any_of(word.begin(), word.end(), [](unsigned char ch) { return std::isdigit(ch) != 0; }) ? 1.0 : 0.0) +
                        (std::isupper(static_cast<unsigned char>(word[0])) ? 2.0 : 0.0));
      }
      // Each question's facts: its capsule's, or, with none, the facts of all capsules that share most with it.
      std::vector<std::vector<std::array<std::size_t, 3>>> asked(questions.size());  // (passage, fact, -)
      std::set<std::size_t> used;
      auto stemsOfFact = [&](std::size_t p, std::size_t f) {
        std::set<std::string> out;
        for (std::size_t k = 0; k < facts[p][f][1]; ++k)
          for (const auto& st : memory::stemsOf(dictionary[wordOf(p, codes[facts[p][f][0] + k])].asString())) out.insert(st);
        return out;
      };
      for (std::size_t i = 0; i < questions.size(); ++i) {
        const double cp = capsuleOf[i];
        if (cp >= 0 && cp < static_cast<double>(passages.size())) {
          const auto p = static_cast<std::size_t>(cp);
          for (std::size_t f = 0; f < std::min<std::size_t>(facts[p].size(), 24); ++f) asked[i].push_back({p, f, 0});
          used.insert(p);
        } else {
          std::set<std::string> qs;
          for (const auto& st : text::words(questions[i][0], true)) qs.insert(st);
          std::vector<std::pair<double, std::array<std::size_t, 3>>> ranked;
          for (std::size_t p = 0; p < passages.size(); ++p)
            for (std::size_t f = 0; f < facts[p].size(); ++f) {
              double shared = 0;
              for (const auto& st : stemsOfFact(p, f)) shared += qs.count(st) ? 1.0 : 0.0;
              ranked.push_back({shared, {p, f, 0}});
            }
          std::stable_sort(ranked.begin(), ranked.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
          for (std::size_t k = 0; k < ranked.size() && k < 24; ++k) {
            asked[i].push_back(ranked[k].second);
            used.insert(ranked[k].second[0]);
          }
        }
      }
      // How informative each stem is, over the facts of the capsules it answers from.
      std::map<std::string, double> df;
      double nFacts = 0;
      for (std::size_t p : used)
        for (std::size_t f = 0; f < facts[p].size(); ++f) {
          for (const auto& st : stemsOfFact(p, f)) df[st] += 1;
          nFacts += 1;
        }
      auto idf = [&](const std::string& w) {
        auto it = df.find(w);
        const double d = it == df.end() ? 0.0 : it->second;
        return std::max(0.0, std::log((nFacts - d + 0.5) / (d + 0.5) + 1.0));
      };
      const memory::Memory mem = memory::recall(!c.flag("memory", true), c.str("memoryFingerprint", ""));
      std::vector<double> params(16, 0.0);
      for (int k = 0; k < 8; ++k) params[5 + static_cast<std::size_t>(k)] = mem.construct[k];
      params[13] = c.num("threshold", 1.0);
      params[0] = double(params.size());
      const auto coords = coordinates(dictionary);
      params.insert(params.end(), coords.begin(), coords.end());
      params[1] = double(params.size());
      params.insert(params.end(), codes.begin(), codes.end());
      params[2] = double(params.size());
      std::vector<double> vocabStart;
      for (const auto& p : passages) {
        vocabStart.push_back(double(params.size()) - params[2]);
        const Value& v = p.at("vocabulary");
        for (std::size_t k = 0; k < v.size(); ++k) params.push_back(v[k].asNumber());
      }
      params[15] = double(params.size());
      params.insert(params.end(), flags.begin(), flags.end());
      // The parts of every question, their asked words, and the facts they answer from.
      std::vector<double> stemRows, forms, factRows;
      std::vector<std::vector<double>> rows;
      Value partsMeta = Value::array();
      for (std::size_t i = 0; i < questions.size(); ++i) {
        const auto parts = memory::questionParts(questions[i][0]);
        for (std::size_t pi = 0; pi < parts.size(); ++pi) {
          const auto& part = parts[pi];
          std::vector<std::string> stems = part.stems;
          for (const auto& r : part.referred) stems.push_back(r);
          std::sort(stems.begin(), stems.end());
          stems.erase(std::unique(stems.begin(), stems.end()), stems.end());
          if (stems.size() > 16) {
            std::stable_sort(stems.begin(), stems.end(), [&](const std::string& a, const std::string& b) { return idf(a) > idf(b); });
            stems.resize(16);
          }
          const double stemFirst = double(stemRows.size() / 3);
          for (const auto& st : stems) {
            const auto& fs = formsOf[st];
            stemRows.insert(stemRows.end(), {idf(st), double(forms.size()), double(fs.size())});
            forms.insert(forms.end(), fs.begin(), fs.end());
          }
          // The facts of one capsule (open questions take their facts in order of what they share).
          const double factFirst = double(factRows.size() / 3);
          Value from = Value::array();
          for (const auto& af : asked[i]) {
            factRows.insert(factRows.end(), {double(facts[af[0]][af[1]][0]), double(facts[af[0]][af[1]][1]), vocabStart[af[0]]});
            from.push(Value::Numbers({double(af[0]), double(af[1])}));
          }
          rows.push_back({factFirst, double(asked[i].size()), 0.0, stemFirst, double(stems.size()),
                          part.asksNumber ? 1.0 : 0.0, part.asksName ? 1.0 : 0.0, double(parts.size())});
          Value referred = Value::array();
          for (const auto& r : part.referred) referred.push(r);
          partsMeta.push(Value::object().set("question", static_cast<double>(i)).set("part", part.text).set("facts", from).set("referred", referred));
        }
      }
      params[3] = double(params.size());
      params.insert(params.end(), stemRows.begin(), stemRows.end());
      params[4] = double(params.size());
      params.insert(params.end(), forms.begin(), forms.end());
      params[14] = double(params.size());
      params.insert(params.end(), factRows.begin(), factRows.end());
      std::uint32_t most = 5;
      for (const auto& r : rows) most = std::max<std::uint32_t>(most, static_cast<std::uint32_t>(r[1]) * 5);
      Plan p = makePlan("capsuleQA", most, 1, params, rows, seedOf(spec), 9);
      p.keepAgents = p.agents;
      Value qs = Value::array();
      for (const auto& q : questions) qs.push(Value::object().set("question", q[0]).set("golden", q[1]));
      p.meta = Value::object().set("questions", qs).set("parts", partsMeta).set("capsule", capsule)
                   .set("memory", Value::object().set("used", mem.used).set("reason", mem.reason).set("generation", mem.generation)
                                      .set("design", mem.constructDesign));
      return std::vector<Plan>{p};
    };
    m.summarise = [](const Value&, const std::vector<Plan>& plans, const std::vector<Output>& out) {
      const Plan& p = plans[0];
      const auto& v = out[0].agentValues;
      const Value& capsule = p.meta.at("capsule");
      const Value& dictionary = capsule.at("dictionary");
      const auto& passages = capsule.at("passages").asArray();
      const char* patterns[5] = {"the whole fact", "its predicate", "its object", "its subject", "the fact and the next"};
      // A fact's words by role, in order of their places (read off the codes).
      auto factWords = [&](std::size_t pi, std::size_t f) {
        const Value& sizes = passages[pi].at("facts");
        std::size_t at = 0;
        for (std::size_t k = 0; k < f; ++k) at += static_cast<std::size_t>(sizes[k].asNumber());
        std::vector<std::array<std::uint32_t, 3>> slots;  // (place, role, word)
        for (std::size_t k = 0; k < static_cast<std::size_t>(sizes[f].asNumber()); ++k) {
          const auto code = static_cast<std::uint32_t>(passages[pi].at("codes")[at + k].asNumber());
          slots.push_back({code % 32u, (code / 32u) % 3u, static_cast<std::uint32_t>(passages[pi].at("vocabulary")[code / 96u].asNumber())});
        }
        std::sort(slots.begin(), slots.end());
        std::array<std::vector<std::string>, 3> roles;
        for (const auto& s : slots) roles[s[1]].push_back(dictionary[s[2]].asString());
        return roles;
      };
      auto join = [](const std::vector<std::string>& ws) {
        std::string s;
        for (const auto& w : ws) s += (s.empty() ? "" : " ") + w;
        return s;
      };
      const auto& questions = p.meta.at("questions").asArray();
      std::vector<Value> traces(questions.size(), Value::array());
      std::vector<std::vector<std::string>> sentences(questions.size());
      for (std::uint32_t j = 0; j < p.jobs; ++j) {
        const Value& part = p.meta.at("parts").asArray()[j];
        const auto qi = static_cast<std::size_t>(part.at("question").asNumber());
        const auto& from = part.at("facts").asArray();
        std::size_t best = 0;
        double belief = 0;
        for (std::size_t a = 0; a < from.size() * 5 && a < p.keepAgents; ++a) {
          const double b = v[(static_cast<std::size_t>(j) * p.keepAgents + a) * 9];
          if (b > belief) {
            belief = b;
            best = a;
          }
        }
        Value trace = Value::object().set("part", part.at("part")).set("referred", part.at("referred"));
        if (belief <= 0) {
          traces[qi].push(trace.set("answer", ""));
          continue;
        }
        const std::size_t f = best / 5, pattern = best % 5;
        const auto pi = static_cast<std::size_t>(from[f][0].asNumber()), fi = static_cast<std::size_t>(from[f][1].asNumber());
        const auto roles = factWords(pi, fi);
        std::vector<std::string> said;
        if (pattern == 0 || pattern == 4) {
          for (const auto& r : roles) said.insert(said.end(), r.begin(), r.end());
        } else if (pattern == 1) {
          said = roles[1];
          said.insert(said.end(), roles[2].begin(), roles[2].end());
        } else {
          said = roles[pattern == 2 ? 2 : 0];
        }
        if (pattern == 4 && f + 1 < from.size()) {
          const auto next = factWords(static_cast<std::size_t>(from[f + 1][0].asNumber()), static_cast<std::size_t>(from[f + 1][1].asNumber()));
          said.push_back(".");
          for (const auto& r : next) said.insert(said.end(), r.begin(), r.end());
        }
        const std::string sentence = memory::realize(said);
        // The reasoning: which habits weighed most for this answer.
        Value why = Value::array();
        std::vector<std::pair<double, int>> parts;
        for (int k = 0; k < 8; ++k)
          parts.push_back({p.params[5 + static_cast<std::size_t>(k)] * v[(static_cast<std::size_t>(j) * p.keepAgents + best) * 9 + 1 + static_cast<std::size_t>(k)], k});
        std::sort(parts.begin(), parts.end(), [](const auto& x, const auto& y) { return x.first > y.first; });
        for (const auto& [c, k] : parts)
          if (c > 0.05) why.push(Value::object().set("habit", memory::kConstructHabits[k]).set("weight", c));
        trace.set("evidence", Value::object().set("subject", join(roles[0])).set("relation", join(roles[1])).set("object", join(roles[2]))
                                  .set("capsule", static_cast<double>(pi)).set("fact", static_cast<double>(fi)))
            .set("saying", patterns[pattern])
            .set("reasoning", why)
            .set("belief", belief)
            .set("answer", sentence);
        traces[qi].push(trace);
        if (std::find(sentences[qi].begin(), sentences[qi].end(), sentence) == sentences[qi].end()) sentences[qi].push_back(sentence);
      }
      Value answers = Value::array();
      double sum = 0, scored = 0;
      for (std::size_t i = 0; i < questions.size(); ++i) {
        std::string answer;
        for (const auto& s : sentences[i]) answer += (answer.empty() ? "" : " ") + s;
        Value a = Value::object().set("question", questions[i].at("question")).set("answer", answer).set("parts", traces[i]);
        const std::string golden = questions[i].at("golden").asString();
        if (!golden.empty()) {
          const double f1 = text::tokenF1(answer, golden);
          a.set("golden", golden).set("f1", f1);
          sum += f1;
          scored += 1;
        }
        answers.push(a);
      }
      Value result = Value::object().set("answers", answers).set("memory", p.meta.at("memory"));
      if (scored > 0) result.set("f1", sum / scored).set("scored", scored);
      return result;
    };
    return m;
  }();
  return &model;
}

}  // namespace syn::models
