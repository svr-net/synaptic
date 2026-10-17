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
// and the answer is the sentence believed most. Two plans: reading (the candidates are the
// sentences of the question's own passage) and open book (every sentence of the corpus, so that the
// reader must also find the passage).
#include <algorithm>
#include <cctype>
#include <cmath>
#include <map>
#include <set>
#include <string>

#include "reading_text.hpp"
#include "syn/models/models.hpp"

// The agent program, in WGSL.
constexpr const char* reading_agent = R"wgsl(
  // Constants: P(0) k1, P(1) b, P(2) τ, P(3) mean sentence length, P(4) start of the idf table,
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
      // Documents: the passages first, then the corpus. Their sentences, tokenised.
      std::vector<std::string> docs;
      for (const auto& p : data.at("passages").asArray()) docs.push_back(p.asString());
      const std::size_t passageCount = docs.size();
      if (data.has("corpus"))
        for (const auto& c : data.at("corpus").asArray()) docs.push_back(c.asString());
      std::map<std::string, int> vocab;
      std::vector<double> df;
      auto id = [&](const std::string& w) {
        auto it = vocab.find(w);
        if (it != vocab.end()) return it->second;
        const int k = static_cast<int>(vocab.size());
        vocab.emplace(w, k);
        df.push_back(0.0);
        return k;
      };
      std::vector<double> pool, table;
      std::vector<int> docFirst, docCount, sentenceDoc;
      Value texts = Value::array();
      double totalLen = 0;
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
      // What the reader learned: how informative each word is (BM25's idf, floored at 0).
      const double nDocs = static_cast<double>(docs.size());
      std::vector<double> idf(df.size());
      for (std::size_t k = 0; k < df.size(); ++k) idf[k] = std::max(0.0, std::log((nDocs - df[k] + 0.5) / (df[k] + 0.5) + 1.0));
      // Questions: their distinct content words that the corpus knows.
      std::vector<std::vector<double>> readingRows, openRows;
      int maxCount = 1;
      Value qmeta = Value::array();
      for (const auto& q : data.at("questions").asArray()) {
        std::set<int> qs;
        for (const auto& w : words(q.at("question").asString(), true)) {
          auto it = vocab.find(w);
          if (it != vocab.end()) qs.insert(it->second);
        }
        const double start = static_cast<double>(pool.size());
        for (int k : qs) pool.push_back(k);
        const auto p = static_cast<std::size_t>(q.at("passage").asNumber());
        if (p >= passageCount) throw std::invalid_argument("readingQA: a question names a passage that is not given");
        readingRows.push_back({start, double(qs.size()), double(docFirst[p]), double(docCount[p])});
        openRows.push_back({start, double(qs.size()), 0.0, double(sentenceDoc.size())});
        maxCount = std::max(maxCount, docCount[p]);
        Value qm = Value::object().set("passage", static_cast<double>(p)).set("question", q.at("question"));
        if (q.has("golden")) qm.set("golden", q.at("golden"));
        qmeta.push(qm);
      }
      std::vector<double> params = {given.num("k1", 1.2), given.num("b", 0.75), given.num("threshold", 4.0),
                                    totalLen / std::max<double>(1.0, static_cast<double>(sentenceDoc.size())), 0, 0, 0, 0};
      params[4] = static_cast<double>(params.size());
      params.insert(params.end(), idf.begin(), idf.end());
      params[5] = static_cast<double>(params.size());
      params.insert(params.end(), pool.begin(), pool.end());
      params[6] = static_cast<double>(params.size());
      params.insert(params.end(), table.begin(), table.end());
      const auto agentsOpen = static_cast<std::uint32_t>(std::max<std::size_t>(1, sentenceDoc.size()));
      Plan reading = makePlan("readingQA", static_cast<std::uint32_t>(maxCount), 1, params, readingRows, seedOf(spec), 1);
      reading.keepAgents = reading.agents;
      Value meta = Value::object();
      Value docOf = Value::array();
      for (int d : sentenceDoc) docOf.push(static_cast<double>(d));
      Value firsts = Value::array();
      for (int f : docFirst) firsts.push(static_cast<double>(f));
      meta.set("sentences", texts).set("doc", docOf).set("docFirst", firsts).set("questions", qmeta).set("vocabulary", static_cast<double>(vocab.size()))
          .set("documents", nDocs).set("passages", static_cast<double>(passageCount));
      reading.meta = meta;
      std::vector<Plan> plans{reading};
      if (given.num("openBook", 1.0) > 0.5) {
        Plan open = makePlan("readingQA", agentsOpen, 1, params, openRows, seedOf(spec), 1);
        open.keepAgents = open.agents;
        plans.push_back(open);
      }
      return plans;
    };
    m.summarise = [](const Value&, const std::vector<Plan>& plans, const std::vector<Output>& out) {
      const Value& meta = plans[0].meta;
      const auto& texts = meta.at("sentences").asArray();
      const auto& docOf = meta.at("doc").asArray();
      auto choose = [&](std::size_t planIndex, std::uint32_t job, std::size_t& index, double& belief) {
        const Plan& p = plans[planIndex];
        const auto& v = out[planIndex].agentValues;
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
      const auto& questions = meta.at("questions").asArray();
      const auto& docFirst = meta.at("docFirst").asArray();
      // Against golden answers, when the questions have them: token F1 of the reader's answers, and
      // of the passage's first sentence, a random sentence and the oracle (its best sentence).
      double sum[5] = {0, 0, 0, 0, 0}, half[5] = {0, 0, 0, 0, 0}, found = 0, scored = 0;
      std::vector<double> bins(20, 0.0);
      Value answers = Value::array();
      for (std::uint32_t j = 0; j < plans[0].jobs; ++j) {
        std::size_t i = 0;
        double b = 0;
        choose(0, j, i, b);
        const Value& q = questions[j];
        Value a = Value::object().set("question", q.at("question")).set("reading", texts[i]).set("readingBelief", b)
                      .set("readingSentence", static_cast<double>(i));
        std::size_t openIndex = i;
        if (plans.size() > 1) {
          choose(1, j, openIndex, b);
          a.set("openBook", texts[openIndex]).set("openBookBelief", b).set("openBookSentence", static_cast<double>(openIndex))
              .set("openBookDocument", docOf[openIndex]);
        }
        if (q.has("golden")) {
          const std::string& g = q.at("golden").asString();
          const auto p = static_cast<std::size_t>(q.at("passage").asNumber());
          const auto first = static_cast<std::size_t>(docFirst[p].asNumber());
          const auto last = p + 1 < docFirst.size() ? static_cast<std::size_t>(docFirst[p + 1].asNumber()) : texts.size();
          double oracle = 0;
          for (std::size_t s = first; s < last; ++s) oracle = std::max(oracle, tokenF1(texts[s].asString(), g));
          const std::size_t pick = first + (last > first ? hash32(j * 2654435761u + 2026u) % (last - first) : 0);
          const double f[5] = {tokenF1(texts[i].asString(), g), tokenF1(texts[openIndex].asString(), g),
                               first < last ? tokenF1(texts[first].asString(), g) : 0.0, first < last ? tokenF1(texts[pick].asString(), g) : 0.0, oracle};
          for (int k = 0; k < 5; ++k) {
            sum[k] += f[k];
            half[k] += f[k] >= 0.5 ? 1.0 : 0.0;
          }
          bins[std::min<std::size_t>(9, static_cast<std::size_t>(f[0] * 10))] += 1;
          bins[10 + std::min<std::size_t>(9, static_cast<std::size_t>(oracle * 10))] += 1;
          if (static_cast<std::size_t>(docOf[openIndex].asNumber()) == p) found += 1;
          scored += 1;
          a.set("golden", g).set("f1", f[0]).set("openBookF1", f[1]).set("oracle", oracle);
        }
        answers.push(a);
      }
      Value result = Value::object()
                         .set("answers", answers)
                         .set("sentences", meta.at("sentences"))
                         .set("doc", meta.at("doc"))
                         .set("docFirst", meta.at("docFirst"))
                         .set("vocabulary", meta.at("vocabulary"))
                         .set("documents", meta.at("documents"))
                         .set("passages", meta.at("passages"));
      if (scored > 0) {
        const char* keys[5] = {"reading", "openBook", "firstSentence", "randomSentence", "oracle"};
        Value scores = Value::object(), atLeastHalf = Value::object();
        for (int k = 0; k < 5; ++k) {
          scores.set(keys[k], sum[k] / scored);
          atLeastHalf.set(keys[k], half[k] / scored);
        }
        scores.set("openBookFoundPassage", found / scored);
        result.set("scores", scores).set("atLeastHalf", atLeastHalf).set("scored", scored)
            .set("readingF1Histogram", Value::Numbers(std::vector<double>(bins.begin(), bins.begin() + 10)))
            .set("oracleF1Histogram", Value::Numbers(std::vector<double>(bins.begin() + 10, bins.end())));
      }
      return result;
    };
    return m;
  }();
  return &model;
}

}  // namespace syn::models
