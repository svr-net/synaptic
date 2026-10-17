// Taking notes: the reader keeps, from each passage it reads, the few sentences it believes are
// worth keeping — before it knows any question — so that it can answer when the passage is gone.
//
// Which sentences are worth keeping is the other half of the reader's memory (reading_memory.hpp):
// note-taking habits learned from golden answers (readingStudent with `notes`) — how informative a
// sentence's words are, how dense, whether it shares words with the rest of the passage, its
// position, length, whether it is a question, whether it holds a number or a name. The belief that
// a sentence is worth keeping is its score read as a point of the space, chance(atLogit(score − τ)).
// Agents are sentences, jobs are passages; the belief of every sentence is kept, and the notes are
// the `keep` sentences believed most (as many as its stored knowledge keeps), in the passage's order. tools/qa_golden writes the notes as
// the reader's knowledge (reader_knowledge.inc), and readingQA answers from them.
#include <algorithm>
#include <numeric>
#include <string>

#include "reading_memory.hpp"
#include "reading_text.hpp"
#include "syn/models/models.hpp"

// The agent program, in WGSL.
constexpr const char* notes_agent = R"wgsl(
  // Constants: P(0..7) note-taking weights, P(8) start of the features (8 per sentence), P(9) τ.
  let first = u32(J(0u));
  let count = u32(J(1u));
  if (A >= count) {
    quant(0u, 0.0);
    return;
  }
  let base = u32(P(8u)) + (first + A) * 8u;
  var score = 0.0;
  for (var k = 0u; k < 8u; k = k + 1u) { score = score + P(k) * P(base + k); }
  // The belief that this sentence is worth keeping: its score as a point of the space.
  let belief = chance(atLogit(clamp(score - P(9u), -30.0, 30.0), kA));
  quant(0u, belief);
  stat(0u, belief);
)wgsl";

namespace syn::models {

namespace {

using memory::Item;

}  // namespace

const kernel::Model* readingNotesModel() {
  static const kernel::Model model = [] {
    kernel::Model m;
    m.name = "readingNotes";
    m.title = "Taking notes: the reader keeps the sentences of each passage it believes are worth keeping, to answer without the passage";
    m.maxStats = 1;
    m.agentSource = notes_agent;
    m.compile = [](const Value& spec) {
      const Value& c = spec.sub("readingNotes");
      std::vector<Item> items;
      if (c.has("passages")) {
        const auto& golden = c.has("golden") ? c.at("golden").asArray() : Value::Array{};
        std::size_t i = 0;
        for (const auto& p : c.at("passages").asArray()) {
          items.push_back({"", i < golden.size() ? golden[i].asString() : std::string(), p.asString()});
          ++i;
        }
      } else {
        const Value g = text::goldenSet(c.integer("questions", 60));
        const auto& ps = g.at("passages").asArray();
        for (const auto& q : g.at("questions").asArray())
          items.push_back({"", q.at("golden").asString(), ps[static_cast<std::size_t>(q.at("passage").asNumber())].asString()});
      }
      const auto cands = memory::noteCandidates(items);
      const memory::Memory mem = memory::recall(!c.flag("memory", true), c.str("memoryFingerprint", ""));
      std::vector<double> params(mem.notes, mem.notes + memory::kHabitCount);
      if (c.has("weights"))
        for (int k = 0; k < memory::kHabitCount && k < static_cast<int>(c.at("weights").size()); ++k)
          params[static_cast<std::size_t>(k)] = c.at("weights")[static_cast<std::size_t>(k)].asNumber();
      params.push_back(0);
      params.push_back(c.num("threshold", 1.0));
      params[8] = static_cast<double>(params.size());
      std::vector<std::vector<double>> rows;
      Value texts = Value::array(), rewards = Value::array(), passageChars = Value::array();
      std::uint32_t maxCount = 1, first = 0;
      for (std::size_t i = 0; i < cands.size(); ++i) {
        const auto& p = cands[i];
        rows.push_back({double(first), double(p.texts.size())});
        Value t = Value::array(), r = Value::array();
        for (std::size_t k = 0; k < p.texts.size(); ++k) {
          params.insert(params.end(), p.habits[k].begin(), p.habits[k].end());
          t.push(p.texts[k]);
          r.push(p.reward[k]);
        }
        texts.push(t);
        rewards.push(r);
        passageChars.push(static_cast<double>(items[i].passage.size()));
        first += static_cast<std::uint32_t>(p.texts.size());
        maxCount = std::max<std::uint32_t>(maxCount, static_cast<std::uint32_t>(p.texts.size()));
      }
      if (rows.empty()) throw std::invalid_argument("readingNotes: no passages");
      Plan p = makePlan("readingNotes", maxCount, 1, params, rows, seedOf(spec), 1);
      p.keepAgents = p.agents;
      Value weights = Value::array(), habits = Value::array();
      for (int k = 0; k < memory::kHabitCount; ++k) {
        weights.push(params[static_cast<std::size_t>(k)]);
        habits.push(memory::kNoteHabits[k]);
      }
      p.meta = Value::object()
                   .set("texts", texts)
                   .set("rewards", rewards)
                   .set("passageChars", passageChars)
                   .set("keep", static_cast<double>(std::max(1, c.integer("keep", memory::knowledge().perPassage > 0 ? memory::knowledge().perPassage : 3))))
                   .set("golden", !items.empty() && !items[0].golden.empty())
                   .set("memory", Value::object().set("used", mem.used).set("reason", mem.reason).set("generation", mem.generation)
                                      .set("design", mem.notesDesign).set("weights", weights).set("habits", habits));
      return std::vector<Plan>{p};
    };
    m.summarise = [](const Value&, const std::vector<Plan>& plans, const std::vector<Output>& out) {
      const Plan& p = plans[0];
      const auto& v = out[0].agentValues;
      const auto keep = static_cast<std::size_t>(p.meta.num("keep", 3));
      const bool golden = p.meta.at("golden").asBool();
      Value notes = Value::array(), passages = Value::array();
      double keptChars = 0, allChars = 0, best = 0, bestAll = 0, sentencesKept = 0, sentencesAll = 0;
      for (std::uint32_t j = 0; j < p.jobs; ++j) {
        const auto& texts = p.meta.at("texts").asArray()[j].asArray();
        const auto& rewards = p.meta.at("rewards").asArray()[j].asArray();
        std::vector<std::size_t> order(texts.size());
        std::iota(order.begin(), order.end(), 0);
        auto belief = [&](std::size_t a) { return static_cast<double>(v[static_cast<std::size_t>(j) * p.keepAgents + a]); };
        std::stable_sort(order.begin(), order.end(), [&](std::size_t x, std::size_t y) { return belief(x) > belief(y); });
        order.resize(std::min(keep, order.size()));
        std::sort(order.begin(), order.end());
        std::string note;
        double kept = 0, all = 0;
        Value sentences = Value::array();
        for (std::size_t a = 0; a < texts.size(); ++a) {
          const bool k = std::find(order.begin(), order.end(), a) != order.end();
          sentences.push(Value::object().set("text", texts[a]).set("belief", belief(a)).set("kept", k));
          all = std::max(all, rewards[a].asNumber());
          if (k) kept = std::max(kept, rewards[a].asNumber());
        }
        for (std::size_t a : order) note += (note.empty() ? "" : " ") + texts[a].asString();
        notes.push(note);
        keptChars += static_cast<double>(note.size());
        allChars += p.meta.at("passageChars").asArray()[j].asNumber();
        sentencesKept += static_cast<double>(order.size());
        sentencesAll += static_cast<double>(texts.size());
        best += kept;
        bestAll += all;
        passages.push(Value::object().set("sentences", sentences));
      }
      Value result = Value::object()
                         .set("notes", notes)
                         .set("passages", passages)
                         .set("keep", static_cast<double>(keep))
                         .set("kept", allChars > 0 ? keptChars / allChars : 0.0)
                         .set("sentencesKept", sentencesKept)
                         .set("sentences", sentencesAll)
                         .set("memory", p.meta.at("memory"));
      // With golden answers: how close the best kept sentence comes to the golden answer, against
      // the best sentence of the whole passage.
      if (golden) result.set("bestKeptF1", best / p.jobs).set("bestF1", bestAll / p.jobs);
      return result;
    };
    return m;
  }();
  return &model;
}

}  // namespace syn::models
