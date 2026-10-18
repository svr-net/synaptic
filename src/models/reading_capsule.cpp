// The reader's knowledge capsule: what it reads, ingested into the space, and recalled from it.
//
// Ingesting (capsuleIngest): the grammar reads every clause of a passage as a fact — subject,
// relation (the verb and what goes with it), object — and each word of a fact becomes a point of
// the space: the point of S³ its coordinates lift to (the same for every reader), turned by the
// grammar's operators — the rotor of its role and the rotor of its place in the fact (bindWord).
// The capsule keeps only these points, each fact's points out of order, and the passage's
// vocabulary (indices into a lexicon of the words it knows); the text is not kept. Agents are the
// slots of a passage (one word each), jobs are passages.
//
// Recalling (capsuleRecall): from the capsule alone, for each point, the word of the passage's
// vocabulary that a rotor of the grammar turns into it, and that rotor — its role and place, read
// off the rotation angles (grammarOf). The facts are put back in order by the rotors alone.
// readingQA and readingStudent construct answers from the recalled facts (reading_memory.hpp),
// never from the text.
#include <algorithm>
#include <array>
#include <map>
#include <set>
#include <string>

#include "reading_memory.hpp"
#include "reading_text.hpp"
#include "syn/models/models.hpp"

// Ingesting, in WGSL.
constexpr const char* ingest_agent = R"wgsl(
  // Constants: P(0) start of the lexicon's coordinates (3 per word), P(1) start of the slots
  // (lexicon index, role, place in the fact). Job: J(0) first slot of the passage, J(1) its slots.
  if (A >= u32(J(1u))) {
    for (var k = 0u; k < 5u; k = k + 1u) { quant(k, 0.0); }
    return;
  }
  let s = u32(J(0u)) + A;
  let word = u32(P(u32(P(1u)) + 3u * s));
  let role = u32(P(u32(P(1u)) + 3u * s + 1u));
  let place = u32(P(u32(P(1u)) + 3u * s + 2u));
  let u = u32(P(0u)) + 3u * word;
  // The word's point of the space, turned by the grammar's rotors of its role and place.
  let x = bindWord(lift(P(u), P(u + 1u), P(u + 2u), 1.0), role, place);
  for (var k = 0u; k < 4u; k = k + 1u) { quant(k, x.x[k]); }
  quant(4u, x.e);
)wgsl";

// Recalling, in WGSL.
constexpr const char* recall_agent = R"wgsl(
  // Constants: P(0) start of the lexicon's coordinates, P(1) start of the vocabularies (lexicon
  // indices), P(2) start of the points (5 per slot). Job: J(0) first slot, J(1) slots, J(2) first
  // word of the passage's vocabulary, J(3) its words.
  if (A >= u32(J(1u))) {
    quant(0u, -1.0);
    return;
  }
  let s = u32(J(0u)) + A;
  let b = u32(P(2u)) + 5u * s;
  let x = V5(vec4<f32>(P(b), P(b + 1u), P(b + 2u), P(b + 3u)), P(b + 4u));
  var found = -1.0;
  // The word of the vocabulary that a rotor of the grammar turns into this point, and the rotor.
  for (var k = 0u; k < u32(J(3u)); k = k + 1u) {
    let word = u32(P(u32(P(1u)) + u32(J(2u)) + k));
    let u = u32(P(0u)) + 3u * word;
    let w = lift(P(u), P(u + 1u), P(u + 2u), 1.0);
    let g = grammarOf(x, w);
    // The rotor read off the angles, checked: it must turn the word's point into this one, exactly.
    if (g >= 0 && dot5(bindWord(w, u32(g) / 32u, u32(g) % 32u), x) > -1e-5) { found = f32(u32(g) * 65536u + word); }
  }
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

}  // namespace

const kernel::Model* capsuleIngestModel() {
  static const kernel::Model model = [] {
    kernel::Model m;
    m.name = "capsuleIngest";
    m.title = "The knowledge capsule: passages read by the grammar, every word a point of the space turned by the rotor of its role";
    m.maxStats = 1;
    m.agentSource = ingest_agent;
    m.helpersWgsl = memory::grammarWgsl();
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
      // The facts of every passage, and the lexicon of the words in them.
      std::vector<std::vector<memory::Fact>> facts;
      std::set<std::string> words;
      double read = 0;
      for (const auto& p : passages) {
        facts.push_back(memory::grammar(p));
        read += static_cast<double>(text::words(p, false).size());
        for (const auto& f : facts.back())
          for (const auto* part : {&f.subject, &f.relation, &f.object}) words.insert(part->begin(), part->end());
      }
      Value dictionary = Value::array();
      std::map<std::string, int> index;
      for (const auto& w : words) {
        index[w] = static_cast<int>(dictionary.size());
        dictionary.push(w);
      }
      std::vector<double> params = {0, 0};
      params[0] = double(params.size());
      const auto coords = coordinates(dictionary);
      params.insert(params.end(), coords.begin(), coords.end());
      params[1] = double(params.size());
      std::vector<std::vector<double>> rows;
      Value shape = Value::array();
      std::size_t first = 0, most = 1;
      for (const auto& fs : facts) {
        std::set<double> vocabulary;
        std::vector<double> sizes;
        std::size_t slots = 0;
        for (const auto& f : fs) {
          const std::vector<std::string>* parts[3] = {&f.subject, &f.relation, &f.object};
          std::vector<std::array<double, 3>> said;  // (word, role, place), in the order of the fact
          for (int r = 0; r < memory::kRoles; ++r)
            for (const auto& w : *parts[r]) said.push_back({double(index[w]), double(r), double(said.size())});
          // Kept out of order (odd places, then even ones backwards): the order is in the rotors.
          std::vector<std::size_t> kept;
          for (std::size_t k = 1; k < said.size(); k += 2) kept.push_back(k);
          for (std::size_t k = said.size(); k-- > 0;)
            if (k % 2 == 0) kept.push_back(k);
          for (std::size_t k : kept) {
            params.insert(params.end(), said[k].begin(), said[k].end());
            vocabulary.insert(said[k][0]);
          }
          sizes.push_back(double(said.size()));
          slots += said.size();
        }
        rows.push_back({double(first), double(slots)});
        shape.push(Value::object().set("vocabulary", Value::Numbers(std::vector<double>(vocabulary.begin(), vocabulary.end()))).set("facts", Value::Numbers(sizes)));
        first += slots;
        most = std::max(most, slots);
      }
      Plan p = makePlan("capsuleIngest", static_cast<std::uint32_t>(most), 1, params, rows, seedOf(spec), 5);
      p.keepAgents = p.agents;
      p.meta = Value::object().set("dictionary", dictionary).set("shape", shape).set("read", read);
      return std::vector<Plan>{p};
    };
    m.summarise = [](const Value&, const std::vector<Plan>& plans, const std::vector<Output>& out) {
      const Plan& p = plans[0];
      const auto& v = out[0].agentValues;
      Value passages = Value::array();
      double slots = 0, facts = 0;
      for (std::uint32_t j = 0; j < p.jobs; ++j) {
        const Value& shape = p.meta.at("shape")[j];
        const auto n = static_cast<std::size_t>(p.jobRows[static_cast<std::size_t>(j) * p.jobStride + 1]);
        std::vector<double> points;
        for (std::size_t a = 0; a < n; ++a)
          for (std::size_t k = 0; k < 5; ++k) points.push_back(v[(static_cast<std::size_t>(j) * p.keepAgents + a) * 5 + k]);
        passages.push(Value::object().set("vocabulary", shape.at("vocabulary")).set("facts", shape.at("facts")).set("points", Value::Numbers(points)));
        slots += static_cast<double>(n);
        facts += static_cast<double>(shape.at("facts").size());
      }
      return Value::object()
          .set("capsule", Value::object().set("dictionary", p.meta.at("dictionary")).set("passages", passages))
          .set("passages", static_cast<double>(p.jobs))
          .set("facts", facts)
          .set("points", slots)
          .set("lexicon", static_cast<double>(p.meta.at("dictionary").size()))
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
    m.title = "Recalling the knowledge capsule: every point turned back by the role rotors to its word and grammatical role";
    m.maxStats = 1;
    m.agentSource = recall_agent;
    m.helpersWgsl = memory::grammarWgsl();
    m.compile = [](const Value& spec) {
      const Value& c = spec.sub("capsuleRecall");
      const Value& capsule = c.has("capsule") ? c.at("capsule") : memory::storedCapsule();
      const Value& dictionary = capsule.at("dictionary");
      const Value& passages = capsule.at("passages");
      if (passages.size() == 0) throw std::invalid_argument("capsuleRecall: an empty capsule");
      std::vector<double> params = {0, 0, 0};
      params[0] = double(params.size());
      const auto coords = coordinates(dictionary);
      params.insert(params.end(), coords.begin(), coords.end());
      params[1] = double(params.size());
      std::vector<double> starts;
      for (std::size_t i = 0; i < passages.size(); ++i) {
        starts.push_back(double(params.size()) - params[1]);
        const Value& vocab = passages.asArray()[i].at("vocabulary");
        for (std::size_t k = 0; k < vocab.size(); ++k) params.push_back(vocab[k].asNumber());
      }
      params[2] = double(params.size());
      std::vector<std::vector<double>> rows;
      std::size_t first = 0, most = 1;
      Value shape = Value::array();
      for (std::size_t i = 0; i < passages.size(); ++i) {
        const Value& points = passages.asArray()[i].at("points");
        for (std::size_t k = 0; k < points.size(); ++k) params.push_back(points[k].asNumber());
        const std::size_t slots = points.size() / 5;
        rows.push_back({double(first), double(slots), starts[i], double(passages.asArray()[i].at("vocabulary").size())});
        shape.push(passages.asArray()[i].at("facts"));
        first += slots;
        most = std::max(most, slots);
      }
      Plan p = makePlan("capsuleRecall", static_cast<std::uint32_t>(most), 1, params, rows, seedOf(spec), 1);
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
        const Value& sizes = p.meta.at("shape")[j];
        Value facts = Value::array();
        std::size_t a = 0;
        for (std::size_t f = 0; f < sizes.size(); ++f) {
          // Each recovered word with the grammar's rotor that placed it: put back in order by the rotors.
          std::vector<std::array<std::uint32_t, 3>> heard;  // (place, role, word)
          for (std::size_t k = 0; k < static_cast<std::size_t>(sizes[f].asNumber()); ++k, ++a) {
            const double code = v[static_cast<std::size_t>(j) * p.keepAgents + a];
            slots += 1;
            if (code < 0) continue;  // not recovered: no word of the vocabulary under any rotor of the grammar
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
