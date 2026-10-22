// Reading for answers: a reader learns which words carry information from a corpus, then answers
// questions with the sentence it believes answers them. The end-to-end QA golden test
// (tools/qa_golden) runs it on FineWeb-Edu passages and golden answers.
//
// What the reader learns from the corpus (every document and passage it is given) is how
// informative each word is: its inverse document frequency. A question asks for its content
// words; each sentence's evidence is the Okapi BM25 weight of the question's words in it (term
// frequency saturating, longer sentences discounted), and the reader's belief that the sentence
// answers the question is that evidence read as a point of the space: chance(atLogit(evidence −
// τ)). Agents are candidate sentences, jobs are questions; the belief of every candidate is kept,
// and the answer is the sentence believed most. Plans: reading (the candidates are the sentences of
// the question's own passage), open book (every sentence of the corpus, so that the reader must also
// find the passage), remembered (the reader's memory: the reading habits it learned as a student
// weigh the passage's sentences and pairs of sentences) and constructed (from its knowledge capsule
// alone: the grammar says answers from the facts recalled from the capsule, and the constructing
// habits it learned as a student choose one; the text is never used — reading_capsule.cpp).
#include <algorithm>
#include <cctype>
#include <cmath>
#include <map>
#include <set>
#include <string>

#include "reading_memory.hpp"
#include "reading_text.hpp"
#include "syn/models/models.hpp"

// The agent program, in WGSL.
constexpr const char* reading_agent = R"wgsl(
  // Remembered habits (J(4) = 1): P(0..7) the habit weights, P(8) start of the candidates' habits
  // (8 per candidate), P(9) τ; the score of a candidate is the weighted sum of its habits.
  if (J(4u) > 0.5) {
    let firstC = u32(J(2u));
    if (A >= u32(J(3u))) {
      quant(0u, 0.0);
      return;
    }
    let base = u32(P(8u)) + (firstC + A) * 8u;
    var score = 0.0;
    for (var k = 0u; k < 8u; k = k + 1u) { score = score + P(k) * P(base + k); }
    let believed = chance(atLogit(clamp(score - P(9u), -30.0, 30.0), kA));
    quant(0u, believed);
    stat(0u, believed);
    return;
  }
  // BM25: P(0) k1, P(1) b, P(2) τ, P(3) mean sentence length, P(4) start of the idf table,
  // P(5) start of the token pool, P(6) start of the sentence table (start, length) pairs.
  let qStart = u32(J(0u));
  let qLen = u32(J(1u));
  let first = u32(J(2u));
  let count = u32(J(3u));
  if (A >= count) {
    quant(0u, 0.0);
    return;
  }
  let s = first + A;
  let start = u32(P(u32(P(6u)) + 2u * s));
  let len = u32(P(u32(P(6u)) + 2u * s + 1u));
  let tokens = u32(P(5u));
  let idf = u32(P(4u));
  let k1 = P(0u);
  let norm = k1 * (1.0 - P(1u) + P(1u) * f32(len) / P(3u));
  var evidence = 0.0;
  for (var i = 0u; i < qLen; i = i + 1u) {
    let t = P(tokens + qStart + i);
    var tf = 0.0;
    for (var k = 0u; k < len; k = k + 1u) {
      if (P(tokens + start + k) == t) { tf = tf + 1.0; }
    }
    if (tf > 0.0) { evidence = evidence + P(idf + u32(t)) * tf * (k1 + 1.0) / (tf + norm); }
  }
  // The belief that this sentence answers the question: the evidence as a point of the space.
  let belief = chance(atLogit(clamp(evidence - P(2u), -30.0, 30.0), kA));
  quant(0u, belief);
  stat(0u, belief);
)wgsl";

namespace syn::models {

namespace {

// The built-in golden set: questions of the QA golden test with their passages and golden answers.
struct Golden {
  const char* question;
  const char* answer;
  const char* passage;
};
const Golden kGolden[] = {
#include "qa_golden_sample.inc"
};

}  // namespace

Value text::goldenSet(int count) {
  Value v = Value::object();
  Value passages = Value::array(), questions = Value::array();
  const int n = std::min<int>(count, static_cast<int>(sizeof(kGolden) / sizeof(kGolden[0])));
  for (int i = 0; i < n; ++i) {
    passages.push(kGolden[i].passage);
    questions.push(Value::object().set("question", kGolden[i].question).set("golden", kGolden[i].answer).set("passage", i));
  }
  v.set("passages", passages).set("questions", questions).set("corpus", Value::array());
  return v;
}

namespace {

// A small built-in example (written for this model) for when no data is given.
Value example() {
  Value v = Value::object();
  Value passages = Value::array(), questions = Value::array(), corpus = Value::array();
  passages.push("Glaciers form where more snow falls in winter than melts in summer. Over many years the snow is pressed into dense ice. "
                "The weight of the ice makes the glacier flow slowly downhill, carving valleys shaped like the letter U.");
  passages.push("Honeybees communicate the location of flowers with a waggle dance. The angle of the dance relative to vertical shows the "
                "direction of the food from the sun. The length of the waggle run tells the other bees how far away the flowers are.");
  questions.push(Value::object().set("question", "What makes a glacier flow downhill?").set("passage", 0));
  questions.push(Value::object().set("question", "How do honeybees show how far away flowers are?").set("passage", 1));
  questions.push(Value::object().set("question", "What is snow pressed into over many years?").set("passage", 0));
  corpus.push("Rivers carve V-shaped valleys as water cuts down into the rock. Floods move sand and gravel downstream.");
  corpus.push("Ants follow scent trails laid by other ants to reach food. A trail fades when no ants renew it.");
  v.set("passages", passages).set("questions", questions).set("corpus", corpus);
  return v;
}

}  // namespace


using namespace text;

namespace {

// Documents read by the reader: their sentences, tokenised, and what it learned from them — how
// informative each word is (BM25's idf, floored at 0).
struct Index {
  std::map<std::string, int> vocab;
  std::vector<double> df, pool, table;
  std::vector<int> docFirst, docCount, sentenceDoc;
  Value texts = Value::array();
  double totalLen = 0, nDocs = 0;

  int id(const std::string& w) {
    auto it = vocab.find(w);
    if (it != vocab.end()) return it->second;
    const int k = static_cast<int>(vocab.size());
    vocab.emplace(w, k);
    df.push_back(0.0);
    return k;
  }
  void read(const std::vector<std::string>& docs) {
    for (std::size_t d = 0; d < docs.size(); ++d) {
      std::set<int> seen;
      docFirst.push_back(static_cast<int>(table.size() / 2));
      int n = 0;
      for (const auto& s : sentences(docs[d])) {
        const auto ws = words(s, true);
        if (ws.empty()) continue;
        table.push_back(static_cast<double>(pool.size()));
        table.push_back(static_cast<double>(ws.size()));
        for (const auto& w : ws) {
          const int k = id(w);
          pool.push_back(k);
          seen.insert(k);
        }
        totalLen += static_cast<double>(ws.size());
        sentenceDoc.push_back(static_cast<int>(d));
        texts.push(s);
        ++n;
      }
      docCount.push_back(n);
      for (int k : seen) df[static_cast<std::size_t>(k)] += 1.0;
    }
    nDocs += static_cast<double>(docs.size());
  }
  // A question: its distinct content words that the reader knows, appended to the pool.
  std::pair<double, double> question(const std::string& q) {
    std::set<int> qs;
    for (const auto& w : words(q, true)) {
      auto it = vocab.find(w);
      if (it != vocab.end()) qs.insert(it->second);
    }
    const double start = static_cast<double>(pool.size());
    for (int k : qs) pool.push_back(k);
    return {start, static_cast<double>(qs.size())};
  }
  std::vector<double> params(const Value& given) const {
    std::vector<double> idf(df.size());
    for (std::size_t k = 0; k < df.size(); ++k) idf[k] = std::max(0.0, std::log((nDocs - df[k] + 0.5) / (df[k] + 0.5) + 1.0));
    std::vector<double> p = {given.num("k1", 1.2), given.num("b", 0.75), given.num("threshold", 4.0),
                             totalLen / std::max<double>(1.0, static_cast<double>(sentenceDoc.size())), 0, 0, 0, 0};
    p[4] = static_cast<double>(p.size());
    p.insert(p.end(), idf.begin(), idf.end());
    p[5] = static_cast<double>(p.size());
    p.insert(p.end(), pool.begin(), pool.end());
    p[6] = static_cast<double>(p.size());
    p.insert(p.end(), table.begin(), table.end());
    return p;
  }
  Value docs() const {
    Value v = Value::array();
    for (int d : sentenceDoc) v.push(static_cast<double>(d));
    return v;
  }
};

Value numbers(const std::vector<int>& xs) {
  Value v = Value::array();
  for (int x : xs) v.push(static_cast<double>(x));
  return v;
}

}  // namespace

const kernel::Model* readingQaModel() {
  static const kernel::Model model = [] {
    kernel::Model m;
    m.name = "readingQA";
    m.title = "Reading for answers: a reader learns word informativeness from a corpus and answers questions (QA golden test)";
    m.maxStats = 1;
    m.agentSource = reading_agent;
    m.compile = [](const Value& spec) {
      const Value& given = spec.sub("readingQA");
      const bool golden = given.str("set", "") == "golden";
      const Value data = golden ? text::goldenSet(given.integer("questions", 60)) : given.has("questions") ? given : example();
      // Documents: the passages first, then the corpus.
      std::vector<std::string> docs;
      // (Index by position: an empty array may come as a numeric array.)
      for (std::size_t i = 0; i < data.at("passages").size(); ++i) docs.push_back(data.at("passages")[i].asString());
      const std::size_t passageCount = docs.size();
      if (data.has("corpus"))
        for (std::size_t i = 0; i < data.at("corpus").size(); ++i) docs.push_back(data.at("corpus")[i].asString());
      // Its knowledge: facts recalled from its capsule (capsuleRecall's result). The constructed
      // answer is said by the grammar from these facts; it never sees the text.
      std::vector<std::vector<memory::Fact>> capsules;
      if (given.has("recalled")) capsules = memory::factsOf(given.at("recalled"));
      const bool construct = !capsules.empty();
      const bool remember = given.num("remember", 1.0) > 0.5 && passageCount > 0;
      const memory::Memory mem = memory::recallFor(given);

      Index main;
      main.read(docs);
      std::vector<std::vector<double>> readingRows, openRows;
      std::vector<memory::Item> items, asked;
      std::vector<std::vector<memory::Fact>> known;
      int maxCount = 1;
      Value qmeta = Value::array();
      for (const auto& q : data.at("questions").asArray()) {
        const std::string text = q.at("question").asString();
        Value qm = Value::object().set("question", text);
        if (q.has("golden")) qm.set("golden", q.at("golden"));
        if (passageCount > 0) {
          const auto [start, len] = main.question(text);
          const auto p = static_cast<std::size_t>(q.num("passage", -1));
          if (!q.has("passage") || p >= passageCount) throw std::invalid_argument("readingQA: a question names a passage that is not given");
          readingRows.push_back({start, len, double(main.docFirst[p]), double(main.docCount[p]), 0});
          openRows.push_back({start, len, 0.0, double(main.sentenceDoc.size()), 0});
          maxCount = std::max(maxCount, main.docCount[p]);
          items.push_back({text, q.str("golden", ""), docs[p]});
          qm.set("passage", static_cast<double>(p));
        }
        if (construct) {
          // The capsule it answers from: the question's own, or else the facts of all that share most with it.
          const double c = q.num("capsule", golden && q.has("passage") ? q.num("passage", -1) : -1);
          known.push_back(c >= 0 && c < static_cast<double>(capsules.size()) ? capsules[static_cast<std::size_t>(c)]
                                                                               : memory::factsFor(text, capsules, 24));
          asked.push_back({text, q.str("golden", ""), ""});
        }
        qmeta.push(qm);
      }
      std::vector<Plan> plans;
      const std::uint32_t seed = seedOf(spec);
      auto withMeta = [](Plan p, const std::string& kind, Value meta) {
        p.keepAgents = p.agents;
        p.meta = meta.set("kind", kind);
        return p;
      };
      if (passageCount > 0) {
        const auto params = main.params(given);
        plans.push_back(withMeta(makePlan("readingQA", static_cast<std::uint32_t>(maxCount), 1, params, readingRows, seed, 1), "reading",
                                 Value::object().set("texts", main.texts).set("doc", main.docs()).set("docFirst", numbers(main.docFirst))));
        if (given.num("openBook", 1.0) > 0.5)
          plans.push_back(withMeta(makePlan("readingQA", static_cast<std::uint32_t>(std::max<std::size_t>(1, main.sentenceDoc.size())), 1, params,
                                            openRows, seed, 1),
                                   "openBook", Value::object().set("texts", main.texts).set("doc", main.docs())));
      }
      if (remember) {
        // The remembered habits weigh the passage's sentences and pairs of sentences.
        const auto cands = memory::answerCandidates(items);
        std::vector<double> params(mem.habits, mem.habits + memory::kHabitCount);
        params.push_back(0);
        params.push_back(given.num("habitThreshold", 1.0));
        params[8] = static_cast<double>(params.size());
        std::vector<std::vector<double>> rows;
        Value texts = Value::array();
        std::size_t first = 0, most = 1;
        for (const auto& c : cands) {
          rows.push_back({0, 0, double(first), double(c.texts.size()), 1});
          for (std::size_t k = 0; k < c.texts.size(); ++k) {
            params.insert(params.end(), c.habits[k].begin(), c.habits[k].end());
            texts.push(c.texts[k]);
          }
          first += c.texts.size();
          most = std::max(most, c.texts.size());
        }
        plans.push_back(withMeta(makePlan("readingQA", static_cast<std::uint32_t>(most), 1, params, rows, seed, 1), "remembered",
                                 Value::object().set("texts", texts)));
      }
      if (construct) {
        // The constructed answers: what the grammar can say from each recalled fact, weighed by the
        // constructing habits it remembers.
        const auto cands = memory::constructCandidates(asked, known);
        std::vector<double> params(mem.construct, mem.construct + memory::kHabitCount);
        params.push_back(0);
        params.push_back(given.num("habitThreshold", 1.0));
        params[8] = static_cast<double>(params.size());
        std::vector<std::vector<double>> rows;
        Value texts = Value::array();
        std::size_t first = 0, most = 1;
        for (const auto& c : cands) {
          rows.push_back({0, 0, double(first), double(c.texts.size()), 1});
          for (std::size_t k = 0; k < c.texts.size(); ++k) {
            params.insert(params.end(), c.habits[k].begin(), c.habits[k].end());
            texts.push(c.texts[k]);
          }
          first += c.texts.size();
          most = std::max(most, c.texts.size());
        }
        if (first == 0) texts.push("");
        plans.push_back(withMeta(makePlan("readingQA", static_cast<std::uint32_t>(most), 1, params, rows, seed, 1), "constructed",
                                 Value::object().set("texts", texts)));
      }
      if (plans.empty()) throw std::invalid_argument("readingQA: neither passages nor a recalled capsule to answer from");
      std::size_t facts = 0;
      for (const auto& c : capsules) facts += c.size();
      plans[0].meta.set("questions", qmeta)
          .set("vocabulary", static_cast<double>(main.vocab.size()))
          .set("documents", main.nDocs)
          .set("passages", static_cast<double>(passageCount))
          .set("memory", [&] {
            Value habits = Value::array();
            for (int k = 0; k < memory::kHabitCount; ++k) habits.push(Value::object().set("habit", memory::kConstructHabits[k]).set("weight", mem.construct[k]));
            return Value::object().set("used", mem.used).set("reason", mem.reason).set("generation", mem.generation).set("design", mem.design)
                .set("constructDesign", mem.constructDesign).set("constructing", habits).set("lessons", mem.lessons).set("records", mem.records);
          }())
          .set("capsule", Value::object().set("used", construct).set("passages", static_cast<double>(capsules.size())).set("facts", static_cast<double>(facts)));
      return plans;
    };
    m.summarise = [](const Value&, const std::vector<Plan>& plans, const std::vector<Output>& out) {
      const Value& meta = plans[0].meta;
      auto planOf = [&](const std::string& kind) -> int {
        for (std::size_t i = 0; i < plans.size(); ++i)
          if (plans[i].meta.at("kind").asString() == kind) return static_cast<int>(i);
        return -1;
      };
      // The candidate believed most for a question: its index in the plan's texts, and the belief.
      auto choose = [&](int planIndex, std::uint32_t job, std::size_t& index, double& belief) {
        const Plan& p = plans[static_cast<std::size_t>(planIndex)];
        const auto& v = out[static_cast<std::size_t>(planIndex)].agentValues;
        const std::uint32_t keep = std::min(p.keepAgents, p.agents);
        const float* row = p.jobRows.data() + static_cast<std::size_t>(job) * p.jobStride;
        const auto count = static_cast<std::uint32_t>(row[3]);
        std::uint32_t best = 0;
        double bestV = -1;
        for (std::uint32_t a = 0; a < std::min(count, keep); ++a) {
          const double b = v[static_cast<std::size_t>(job) * keep + a];
          if (b > bestV) {
            bestV = b;
            best = a;
          }
        }
        index = static_cast<std::size_t>(row[2]) + best;
        belief = bestV;
      };
      const int reading = planOf("reading"), open = planOf("openBook"), remembered = planOf("remembered"), constructed = planOf("constructed");
      const auto& questions = meta.at("questions").asArray();
      // Against golden answers, when the questions have them: token F1 of the reader's answers, and
      // of the passage's first sentence, a random sentence and the oracle (its best sentence).
      const char* keys[7] = {"reading", "openBook", "firstSentence", "randomSentence", "oracle", "remembered", "constructed"};
      double sum[7] = {}, half[7] = {}, have[7] = {}, found = 0, scored = 0;
      std::vector<double> bins(20, 0.0);
      Value answers = Value::array();
      const std::uint32_t jobs = plans[0].jobs;
      for (std::uint32_t j = 0; j < jobs; ++j) {
        const Value& q = questions[j];
        Value a = Value::object().set("question", q.at("question"));
        std::string text[7];
        bool got[7] = {};
        auto answer = [&](int plan, int slot, const char* key) {
          if (plan < 0) return std::size_t(0);
          std::size_t i = 0;
          double b = 0;
          choose(plan, j, i, b);
          const Plan& pl = plans[static_cast<std::size_t>(plan)];
          // No candidate at all (nothing it knows says anything): no answer.
          const bool none = pl.jobRows[static_cast<std::size_t>(j) * pl.jobStride + 3] < 0.5;
          text[slot] = none ? std::string() : pl.meta.at("texts")[i].asString();
          if (none) b = 0;
          got[slot] = true;
          a.set(key, text[slot]).set(std::string(key) + "Belief", b);
          return i;
        };
        const std::size_t ri = answer(reading, 0, "reading");
        const std::size_t oi = answer(open, 1, "openBook");
        answer(remembered, 5, "remembered");
        answer(constructed, 6, "constructed");
        if (reading >= 0) a.set("readingSentence", static_cast<double>(ri));
        if (open >= 0) a.set("openBookSentence", static_cast<double>(oi)).set("openBookDocument", plans[static_cast<std::size_t>(open)].meta.at("doc")[oi]);
        if (q.has("golden")) {
          const std::string& g = q.at("golden").asString();
          if (reading >= 0) {
            const Value& rm = plans[static_cast<std::size_t>(reading)].meta;
            const auto& texts = rm.at("texts").asArray();
            const auto& docFirst = rm.at("docFirst").asArray();
            const auto p = static_cast<std::size_t>(q.at("passage").asNumber());
            const auto first = static_cast<std::size_t>(docFirst[p].asNumber());
            const auto last = p + 1 < docFirst.size() ? static_cast<std::size_t>(docFirst[p + 1].asNumber()) : texts.size();
            double oracle = 0;
            for (std::size_t s = first; s < last; ++s) oracle = std::max(oracle, tokenF1(texts[s].asString(), g));
            const std::size_t pick = first + (last > first ? hash32(j * 2654435761u + 2026u) % (last - first) : 0);
            if (first < last) {
              text[2] = texts[first].asString();
              text[3] = texts[pick].asString();
              got[2] = got[3] = true;
            }
            sum[4] += oracle;
            half[4] += oracle >= 0.5 ? 1.0 : 0.0;
            have[4] += 1;
            bins[10 + std::min<std::size_t>(9, static_cast<std::size_t>(oracle * 10))] += 1;
            a.set("oracle", oracle);
            if (open >= 0 && static_cast<std::size_t>(plans[static_cast<std::size_t>(open)].meta.at("doc")[oi].asNumber()) == p) found += 1;
          }
          for (int k = 0; k < 7; ++k) {
            if (k == 4 || !got[k]) continue;
            const double f = tokenF1(text[k], g);
            sum[k] += f;
            half[k] += f >= 0.5 ? 1.0 : 0.0;
            have[k] += 1;
            if (k == 0) {
              bins[std::min<std::size_t>(9, static_cast<std::size_t>(f * 10))] += 1;
              a.set("f1", f);
            }
            if (k == 1) a.set("openBookF1", f);
            if (k == 5) a.set("rememberedF1", f);
            if (k == 6) a.set("constructedF1", f);
          }
          scored += 1;
          a.set("golden", g);
        }
        answers.push(a);
      }
      std::vector<double> f1s;
      for (const auto& a : answers.asArray()) f1s.push_back(a.num("f1", 0));
      Value result = Value::object()
                         .set("answers", answers)
                         .set("ranked", Value::Numbers(rankedBy(f1s)))
                         .set("vocabulary", meta.at("vocabulary"))
                         .set("documents", meta.at("documents"))
                         .set("passages", meta.at("passages"))
                         .set("memory", meta.at("memory"))
                         .set("capsule", meta.at("capsule"));
      if (reading >= 0) {
        const Value& rm = plans[static_cast<std::size_t>(reading)].meta;
        result.set("sentences", rm.at("texts")).set("doc", rm.at("doc")).set("docFirst", rm.at("docFirst"));
      }
      if (scored > 0) {
        Value scores = Value::object(), atLeastHalf = Value::object();
        for (int k = 0; k < 7; ++k) {
          if (have[k] == 0) continue;
          scores.set(keys[k], sum[k] / have[k]);
          atLeastHalf.set(keys[k], half[k] / have[k]);
        }
        if (open >= 0 && reading >= 0) scores.set("openBookFoundPassage", found / scored);
        result.set("scores", scores).set("atLeastHalf", atLeastHalf).set("scored", scored);
        if (reading >= 0)
          result.set("readingF1Histogram", Value::Numbers(std::vector<double>(bins.begin(), bins.begin() + 10)))
              .set("oracleF1Histogram", Value::Numbers(std::vector<double>(bins.begin() + 10, bins.end())));
      }
      return result;
    };
    return m;
  }();
  return &model;
}

}  // namespace syn::models
