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
fn cueOf(t : u32) -> u32 { return u32(Q(u32(Q(1u)) + 7u * t + 3u)); }
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
  // Constants (after the sign table): Q(0) start of the index's points (3 coordinates per word of
  // the lexicon), Q(1) start of the words (coordinates, cues, first word of its clause, clause
  // length, fact), Q(2) start of the index's cells (ixCells + 1 offsets, then the entries), Q(3..8)
  // the word classes. Job: J(0) first word of the passage, J(1) its words.
  if (A >= u32(J(1u))) {
    quant(0u, -1.0);
    quant(1u, -1.0);
    quant(2u, -1.0);
    return;
  }
  let t = u32(J(0u)) + A;
  let b = u32(Q(1u)) + 7u * t;
  let first = u32(Q(b + 4u));
  let n = u32(Q(b + 5u));
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
    quant(1u, -1.0);
    quant(2u, -1.0);
    return;
  }
  // The slot: grade 0 its fact, grade 1 its word's point, grade 2 the grammar's bivector.
  mvVector(0u, V5(ixLift(vec3<f32>(Q(b), Q(b + 1u), Q(b + 2u))), 1.0));
  mvSet(0u, 0u, Q(b + 6u));
  mvSet(0u, 3u, 0.5 * (0.5 + kGrammarRole * f32(role)));
  mvSet(0u, 12u, 0.5 * (0.3 + kGrammarStep * f32(place)));
  // The index router. Level 1: grade 1 isolated, routed through the growing index — the cell of S³
  // it lies in (the sides of the spheres), then the cell's entry it coincides with; none: a new word.
  mvGrade(1u, 0u, 1u);
  let x = vec4<f32>(mvGet(1u, 1u), mvGet(1u, 2u), mvGet(1u, 4u), mvGet(1u, 8u));
  let xe = mvGet(1u, 16u);
  let cell = ixCell(x, xe);
  let cells = u32(Q(2u));
  var word = -1.0;
  for (var k = u32(Q(cells + cell)); k < u32(Q(cells + cell + 1u)); k = k + 1u) {
    let id = u32(Q(cells + ixCells + 1u + k));
    let v = u32(Q(0u)) + 3u * id;
    if (word < 0.0 && ixCoincide(ixLift(vec3<f32>(Q(v), Q(v + 1u), Q(v + 2u))), 1.0, x, xe)) { word = f32(id); }
  }
  // Level 2: grade 2 isolated, routed to the grammar's lattice of roles and places.
  mvGrade(2u, 0u, 2u);
  let r = u32(clamp(round((2.0 * mvGet(2u, 3u) - 0.5) / kGrammarRole), 0.0, 2.0));
  let p = u32(clamp(round((2.0 * mvGet(2u, 12u) - 0.3) / kGrammarStep), 0.0, 31.0));
  // Level 0: grade 0 isolated — the fact, which the capsule keeps as runs of slots.
  mvGrade(3u, 0u, 0u);
  stat(0u, select(0.0, 1.0, abs(mvGet(3u, 0u) - Q(b + 6u)) < 0.5 && r == role && p == place));
  quant(0u, word);
  quant(1u, f32(r * 32u + p));
  quant(2u, f32(cell));
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

constexpr std::size_t kIndexCells = 4096;  // ixCells

// The growing index's geometry, carried from svr-net/compute (tools/sync_compute.sh).
constexpr const char* compute_index =
#include "compute_index.inc"
    ;

}  // namespace

const kernel::Model* capsuleIngestModel() {
  static const kernel::Model model = [] {
    kernel::Model m;
    m.name = "capsuleIngest";
    m.title = "The knowledge capsule: passages read by the grammar, every word a multivector routed grade by grade to a packed code";
    m.maxStats = 1;
    m.agentSource = ingest_agent;
    m.helpersWgsl = helpers() + word_classes + compute_index;
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
      // The capsule it grows: its `base` segments (on top of the stored capsule when they extend it),
      // the stored capsule with `stored`, or none.
      Value base = Value::object().set("capsule", Value::object().set("dictionary", Value::array()).set("cells", Value::Numbers{}).set("passages", Value::array()))
                       .set("segments", Value::array()).set("stored", false);
      if (c.has("base") && c.at("base").size() > 0)
        base = memory::capsuleOf(Value::object().set("segments", c.at("base")).set("stored", c.has("stored") && c.at("stored").asBool()));
      else if (c.has("stored") && c.at("stored").asBool())
        base = memory::capsuleOf(Value::object());
      const Value& known = base.at("capsule").at("dictionary");
      const Value& knownCells = base.at("capsule").at("cells");
      // The clauses of every passage, with the cues of their words.
      std::vector<std::vector<memory::Clause>> clauses;
      double read = 0, textBytes = 0;
      for (const auto& p : passages) {
        clauses.push_back(memory::clauses(p));
        read += static_cast<double>(text::words(p, false).size());
        textBytes += static_cast<double>(p.size());
      }
      // The word classes: auxiliary, verb ending, each sign against, goes with a verb, after a plural
      // first word, and the chance the verb class's cap holds.
      std::vector<double> q = {0, 0, 0, c.num("auxiliary", 20), c.num("verbEnding", 2), c.num("against", 4), c.num("goesWithVerb", 2),
                               c.num("afterPlural", 2), c.num("verbCap", 0.75)};
      // The index: the points of the words the capsule knows.
      q[0] = double(q.size());
      const auto coords = coordinates(known);
      q.insert(q.end(), coords.begin(), coords.end());
      q[1] = double(q.size());
      std::vector<std::vector<double>> rows;
      Value words = Value::array();
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
            float u[3];
            memory::wordCoordinates(cl.words[k], u);
            q.insert(q.end(), {u[0], u[1], u[2], double(cl.cues[k]), double(clauseFirst), double(cl.words.size()), double(fi)});
            words.push(cl.words[k]);
          }
          lengths.push(static_cast<double>(cl.words.size()));
        }
        firsts.push_back(double(start));
        clauseLengths.push_back(lengths);
        most = std::max(most, token - start);
      }
      // The index's cells: each cell's words, in the order they came.
      q[2] = double(q.size());
      std::vector<std::vector<double>> members(kIndexCells);
      for (std::size_t w = 0; w < known.size(); ++w) members[static_cast<std::size_t>(knownCells[w].asNumber()) % kIndexCells].push_back(double(w));
      double at = 0;
      for (const auto& cell : members) {
        q.push_back(at);
        at += double(cell.size());
      }
      q.push_back(at);
      for (const auto& cell : members) q.insert(q.end(), cell.begin(), cell.end());
      for (std::size_t p = 0; p < clauses.size(); ++p) {
        const double count = p + 1 < clauses.size() ? firsts[p + 1] - firsts[p] : double(token) - firsts[p];
        rows.push_back({firsts[p], count});
      }
      Plan p = makePlan("capsuleIngest", static_cast<std::uint32_t>(most), 1, space::withSignTable(q), rows, seedOf(spec), 3);
      p.keepAgents = p.agents;
      Value shape = Value::array();
      for (const auto& l : clauseLengths) shape.push(l);
      Value given = Value::array();
      if (c.has("base"))
        for (std::size_t i = 0; i < c.at("base").size(); ++i) given.push(c.at("base")[i].asString());
      p.meta = Value::object().set("base", base).set("given", given).set("words", words).set("shape", shape).set("read", read).set("textBytes", textBytes);
      return std::vector<Plan>{p};
    };
    m.summarise = [](const Value&, const std::vector<Plan>& plans, const std::vector<Output>& out) {
      const Plan& p = plans[0];
      const auto& v = out[0].agentValues;
      const Value& baseCapsule = p.meta.at("base").at("capsule");
      const Value& known = baseCapsule.at("dictionary");
      const Value& words = p.meta.at("words");
      const std::size_t baseWords = known.size(), basePassages = baseCapsule.at("passages").size();
      auto value = [&](std::uint32_t j, std::size_t a, std::size_t k) { return v[(static_cast<std::size_t>(j) * p.keepAgents + a) * 3 + k]; };
      // The words new to the index (routed to no entry), with the cell each lies in: numbered after the
      // lexicon, in order (front-coded together). A word the index routed differently than it reads
      // counts against `routed`.
      std::map<std::string, double> fresh;
      std::map<std::string, double> knownAt;
      for (std::size_t w = 0; w < baseWords; ++w) knownAt[known[w].asString()] = double(w);
      double slots = 0, routed = 0;
      std::size_t token = 0;
      for (std::uint32_t j = 0; j < p.jobs; ++j) {
        const Value& lengths = p.meta.at("shape").asArray()[j];
        std::size_t a = 0;
        for (std::size_t f = 0; f < lengths.size(); ++f)
          for (std::size_t k = 0; k < static_cast<std::size_t>(lengths[f].asNumber()); ++k, ++a, ++token) {
            if (value(j, a, 1) < 0) continue;
            const std::string word = words[token].asString();
            const double id = value(j, a, 0);
            const auto it = knownAt.find(word);
            slots += 1;
            routed += (id < 0 ? it == knownAt.end() : it != knownAt.end() && it->second == id) ? 1.0 : 0.0;
            if (it == knownAt.end()) fresh.emplace(word, value(j, a, 2));
          }
      }
      Value dictionary = known, cells = baseCapsule.at("cells");
      std::map<std::string, double> index = knownAt;
      for (const auto& [word, cell] : fresh) {
        index[word] = double(dictionary.size());
        dictionary.push(word);
        cells.numbers().push_back(cell);
      }
      // Each passage: its vocabulary (level 1's codebook) and the codes of its facts.
      Value passages = baseCapsule.at("passages");
      double facts = 0, kept = 0;
      token = 0;
      for (std::uint32_t j = 0; j < p.jobs; ++j) {
        const Value& lengths = p.meta.at("shape").asArray()[j];
        std::vector<std::vector<std::array<double, 2>>> fs;  // (lexicon index, role · 32 + place)
        std::set<double> used;
        std::size_t a = 0;
        for (std::size_t f = 0; f < lengths.size(); ++f) {
          std::vector<std::array<double, 2>> fact;
          for (std::size_t k = 0; k < static_cast<std::size_t>(lengths[f].asNumber()); ++k, ++a, ++token) {
            const double grammar = value(j, a, 1);
            if (grammar < 0) continue;
            const double w = index.at(words[token].asString());
            fact.push_back({w, grammar});
            used.insert(w);
          }
          fs.push_back(fact);
        }
        const std::vector<double> vocabulary(used.begin(), used.end());
        std::vector<double> sizes, codes;
        for (const auto& fact : fs) {
          // Kept out of order (odd places, then even ones backwards): the order is in the grammar's bivectors.
          std::vector<std::size_t> order;
          for (std::size_t k = 1; k < fact.size(); k += 2) order.push_back(k);
          for (std::size_t k = fact.size(); k-- > 0;)
            if (k % 2 == 0) order.push_back(k);
          for (std::size_t k : order) {
            const auto rank = static_cast<double>(std::lower_bound(vocabulary.begin(), vocabulary.end(), fact[k][0]) - vocabulary.begin());
            codes.push_back(rank * 96 + fact[k][1]);
          }
          sizes.push_back(double(fact.size()));
        }
        passages.push(Value::object().set("vocabulary", Value::Numbers(vocabulary)).set("facts", Value::Numbers(sizes)).set("codes", Value::Numbers(codes)));
        kept += static_cast<double>(codes.size());
        facts += static_cast<double>(sizes.size());
      }
      const Value capsule = Value::object().set("dictionary", dictionary).set("cells", cells).set("passages", passages);
      const memory::CapsuleBytes packed = memory::packSegment(capsule, baseWords, basePassages);
      Value segments = p.meta.at("given");
      segments.push(packed.base64);
      const double text = p.meta.num("textBytes", 0);
      return Value::object()
          .set("capsule", capsule)
          .set("segment", packed.base64)
          .set("packed", packed.base64)
          .set("segments", segments)
          .set("base", Value::object()
                           .set("words", static_cast<double>(baseWords))
                           .set("passages", static_cast<double>(basePassages))
                           .set("stored", p.meta.at("base").at("stored"))
                           .set("segments", p.meta.at("base").at("segments")))
          .set("bytes", Value::object()
                            .set("header", static_cast<double>(packed.header))
                            .set("lexicon", static_cast<double>(packed.lexicon))
                            .set("cells", static_cast<double>(packed.cells))
                            .set("vocabulary", static_cast<double>(packed.vocabulary))
                            .set("facts", static_cast<double>(packed.facts))
                            .set("codes", static_cast<double>(packed.codes))
                            .set("total", static_cast<double>(packed.bytes.size()))
                            .set("text", text)
                            .set("ratio", text > 0 ? static_cast<double>(packed.bytes.size()) / text : 0.0))
          .set("routed", slots > 0 ? routed / slots : 0.0)
          .set("grades", [&] {
            // The share of slots whose grades 0 and 2 routed back to their fact, role and place exactly.
            double sum = 0;
            for (std::uint32_t j = 0; j < p.jobs; ++j) sum += out[0].mean(p, j, 0) * p.agents;
            return slots > 0 ? sum / slots : 0.0;
          }())
          .set("passages", static_cast<double>(p.jobs))
          .set("facts", facts)
          .set("points", kept)
          .set("lexicon", static_cast<double>(dictionary.size()))
          .set("newWords", static_cast<double>(dictionary.size() - baseWords))
          .set("knownWords", static_cast<double>(baseWords))
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
      const Value source = memory::capsuleOf(c);
      const Value& capsule = source.at("capsule");
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
      p.meta = Value::object().set("dictionary", dictionary).set("shape", shape).set("segments", source.at("segments")).set("stored", source.at("stored"));
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
          .set("segments", p.meta.at("segments"))
          .set("stored", p.meta.at("stored"))
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
  // of the facts (first slot, slots, first word of its capsule's vocabulary, whether the next fact of
  // its passage follows it), P(15) start of the
  // lexicon's flags (1 a digit, 2 a capital, 4 says what something is: a form of "be", "means", "is called"; 8 goes with
  // a verb without being one: "has", "not", "also", "as", "of"; 16 a past participle: "found", "contained"). Job (a part of a question): J(0) its first fact,
  // J(1) facts, J(3) first asked word, J(4) asked words, J(5) asks a number, J(6) asks a name,
  // J(7) the question's parts, J(8) whether it asks what something is; P(16) the weight of a fact that says what
  // something is, for such a part.
  let nf = u32(J(1u));
  let f = A / 5u;
  let pattern = A % 5u;
  // One sentence per part: a question in parts says one fact for each (Claude's habit 4).
  // The fact and the next: only when the next fact of its passage follows it.
  if (f >= nf || (pattern == 4u && (f + 1u >= nf || J(7u) > 1.5 || P(u32(P(14u)) + 4u * (u32(J(0u)) + f) + 3u) < 0.5))) {
    for (var k = 0u; k < 10u; k = k + 1u) { quant(k, 0.0); }
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
  var copula = false;   // the fact's relation is a form of "be" (or "means", "is called")
  var otherVerb = false;  // ... and no other verb or modal ("DNA can be found in" does not say what DNA is)
  var objectPlace = 99u;  // the object's first word, and whether it is a participle ("is contained in")
  var passive = false;
  var subjectWords = 0u;  // the fact's subject: its words, and the asked words it holds
  var bySubject : array<bool, 16>;
  for (var t = 0u; t < 16u; t = t + 1u) { bySubject[t] = false; }
  var firstPlace = 99u;
  for (var stage = 0u; stage < 2u; stage = stage + 1u) {
    for (var which = 0u; which < select(1u, 2u, pattern == 4u); which = which + 1u) {
      let fact = u32(P(14u)) + 4u * (u32(J(0u)) + f + which);
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
        if (which == 0u && role == 1u && (u32(P(u32(P(15u)) + word)) & 4u) != 0u) { copula = true; }
        if (which == 0u && role == 1u && (u32(P(u32(P(15u)) + word)) & 12u) == 0u) { otherVerb = true; }
        if (which == 0u && role == 2u && place < objectPlace) {
          objectPlace = place;
          passive = (u32(P(u32(P(15u)) + word)) & 16u) != 0u;
        }
        if (which == 0u && role == 0u) { subjectWords = subjectWords + 1u; }
        // Evidence: the question's words in the fact, by identity of their points.
        for (var t = 0u; t < asked; t = t + 1u) {
          let row = u32(P(3u)) + 3u * (u32(J(3u)) + t);
          for (var l = 0u; l < u32(P(row + 2u)); l = l + 1u) {
            let o = u32(P(0u)) + 3u * u32(P(u32(P(4u)) + u32(P(row + 1u)) + l));
            if (dot5(w, lift(P(o), P(o + 1u), P(o + 2u), 1.0)) > -1e-5) {
              if (which == 0u) { held[t] = true; }
              if (which == 0u && role == 0u) { bySubject[t] = true; }
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
    for (var k = 0u; k < 10u; k = k + 1u) { quant(k, 0.0); }
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
  // The fact says what the asked thing is: its relation a form of "be" and no other verb (nor
  // a passive), its subject the asked words (all of them,
  // and little else).
  var inSubject = 0u;
  for (var t = 0u; t < asked; t = t + 1u) { inSubject = inSubject + select(0u, 1u, bySubject[t]); }
  let defines = copula && !otherVerb && !passive && asked > 0u && inSubject == asked && subjectWords <= asked + 1u;
  // Asked what something is, the fact that says what it is weighs more.
  score = score + select(0.0, P(16u), J(8u) > 0.5 && defines);
  // The belief that this is the answer: a point of the space.
  quant(0u, chance(atLogit(clamp(score - P(13u), -30.0, 30.0), kA)));
  quant(9u, select(0.0, 1.0, defines));
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
      const Value source = memory::capsuleOf(c);
      const Value& capsule = source.at("capsule");
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
        static const std::set<std::string> saysWhat = {"is", "are", "was", "were", "be", "been", "being", "means", "mean", "meant",
                                                        "refers", "called", "known", "defined", "consists", "describes"};
        static const std::set<std::string> withVerb = {"has", "have", "had", "do", "does", "did", "not", "also", "often", "usually",
                                                       "still", "only", "then", "as", "of", "to", "by", "in", "for", "with", "than"};
        static const std::set<std::string> irregular = {"found", "made", "given", "seen", "taken", "shown", "done", "held", "kept", "built",
                                                        "written", "born", "grown", "thrown", "hidden", "broken", "chosen", "spoken",
                                                        "frozen", "driven", "eaten", "fallen", "brought", "bought", "caught", "taught",
                                                        "thought", "sent", "spent", "left", "lost", "paid", "said", "sold", "told", "worn"};
        auto participle = [&](const std::string& l) {
          return irregular.count(l) > 0 || (l.size() > 4 && l.compare(l.size() - 2, 2, "ed") == 0 && l.compare(l.size() - 3, 3, "eed") != 0);
        };
        std::string lw = word;
        std::transform(lw.begin(), lw.end(), lw.begin(), [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
        flags.push_back((std::any_of(word.begin(), word.end(), [](unsigned char ch) { return std::isdigit(ch) != 0; }) ? 1.0 : 0.0) +
                        (std::isupper(static_cast<unsigned char>(word[0])) ? 2.0 : 0.0) + (saysWhat.count(lw) ? 4.0 : 0.0) +
                        (!saysWhat.count(lw) && withVerb.count(lw) ? 8.0 : 0.0) + (participle(lw) ? 16.0 : 0.0));
      }
      // Each question's facts: its capsule's, or, with none, the facts of all capsules that share most with it.
      std::vector<std::vector<std::array<std::size_t, 3>>> asked(questions.size());  // (passage, fact, -)
      std::set<std::size_t> used;
      // A fact's stems, read once (open questions rank every fact of every capsule).
      std::vector<std::vector<std::set<std::string>>> factStems(passages.size());
      std::vector<std::vector<bool>> factStemsRead(passages.size());
      auto stemsOfFact = [&](std::size_t p, std::size_t f) -> const std::set<std::string>& {
        if (factStems[p].empty()) {
          factStems[p].resize(facts[p].size());
          factStemsRead[p].assign(facts[p].size(), false);
        }
        if (!factStemsRead[p][f]) {
          for (std::size_t k = 0; k < facts[p][f][1]; ++k)
            for (const auto& st : memory::stemsOf(dictionary[wordOf(p, codes[facts[p][f][0] + k])].asString())) factStems[p][f].insert(st);
          factStemsRead[p][f] = true;
        }
        return factStems[p][f];
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
      const memory::Memory mem = memory::recallFor(c);
      std::vector<double> params(17, 0.0);
      // A part asking what something is prefers the fact that says what something is (its relation a form of "be").
      params[16] = c.num("definitionWeight", 0.0);  // the reader (tools/reader) weighs it; readingQA does not
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
          const double factFirst = double(factRows.size() / 4);
          Value from = Value::array();
          for (std::size_t k = 0; k < asked[i].size(); ++k) {
            const auto& af = asked[i][k];
            const bool followed = k + 1 < asked[i].size() && asked[i][k + 1][0] == af[0] && asked[i][k + 1][1] == af[1] + 1;
            factRows.insert(factRows.end(), {double(facts[af[0]][af[1]][0]), double(facts[af[0]][af[1]][1]), vocabStart[af[0]], followed ? 1.0 : 0.0});
            from.push(Value::Numbers({double(af[0]), double(af[1])}));
          }
          rows.push_back({factFirst, double(asked[i].size()), 0.0, stemFirst, double(stems.size()),
                          part.asksNumber ? 1.0 : 0.0, part.asksName ? 1.0 : 0.0, double(parts.size()), part.asksDefinition ? 1.0 : 0.0});
          Value referred = Value::array();
          for (const auto& r : part.referred) referred.push(r);
          partsMeta.push(Value::object().set("question", static_cast<double>(i)).set("part", part.text).set("facts", from).set("referred", referred)
                             .set("definition", part.asksDefinition));
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
      Plan p = makePlan("capsuleQA", most, 1, params, rows, seedOf(spec), 10);
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
          const double b = v[(static_cast<std::size_t>(j) * p.keepAgents + a) * 10];
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
          parts.push_back({p.params[5 + static_cast<std::size_t>(k)] * v[(static_cast<std::size_t>(j) * p.keepAgents + best) * 10 + 1 + static_cast<std::size_t>(k)], k});
        std::sort(parts.begin(), parts.end(), [](const auto& x, const auto& y) { return x.first > y.first; });
        for (const auto& [c, k] : parts)
          if (c > 0.05) why.push(Value::object().set("habit", memory::kConstructHabits[k]).set("weight", c));
        trace.set("evidence", Value::object().set("subject", join(roles[0])).set("relation", join(roles[1])).set("object", join(roles[2]))
                                  .set("capsule", static_cast<double>(pi)).set("fact", static_cast<double>(fi)))
            .set("holds", v[(static_cast<std::size_t>(j) * p.keepAgents + best) * 10 + 1])
            .set("covers", v[(static_cast<std::size_t>(j) * p.keepAgents + best) * 10 + 8])
            .set("defines", v[(static_cast<std::size_t>(j) * p.keepAgents + best) * 10 + 9] > 0.5)
            .set("definition", part.at("definition"))
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
