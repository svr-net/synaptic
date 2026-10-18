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
