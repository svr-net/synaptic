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

const std::set<std::string>& stopwords() {
  static const std::set<std::string> s = {
      "a", "an", "the", "and", "or", "but", "if", "of", "to", "in", "on", "at", "by", "for", "with", "from", "as", "into", "about", "than",
      "is", "are", "was", "were", "be", "been", "being", "am", "do", "does", "did", "has", "have", "had", "it", "its", "this", "that",
      "these", "those", "there", "their", "they", "them", "he", "she", "his", "her", "him", "we", "our", "you", "your", "i", "my", "me",
      "what", "which", "who", "whom", "whose", "when", "where", "why", "how", "can", "could", "would", "should", "will", "may", "might",
      "must", "not", "no", "so", "such", "also", "some", "any", "all", "each", "other", "more", "most", "one", "two", "according",
      "passage", "text", "does", "do", "s", "t"};
  return s;
}

// Lower-case words of letters and digits, with a light stemming of plural and verb endings.
std::vector<std::string> words(const std::string& text, bool content) {
  std::vector<std::string> out;
  std::string w;
  auto flush = [&]() {
    if (w.empty()) return;
    if (w.size() > 4 && w.compare(w.size() - 3, 3, "ing") == 0) w.resize(w.size() - 3);
    else if (w.size() > 3 && w.compare(w.size() - 2, 2, "ed") == 0) w.resize(w.size() - 2);
    else if (w.size() > 3 && w.compare(w.size() - 2, 2, "es") == 0) w.resize(w.size() - 2);
    else if (w.size() > 3 && w.back() == 's') w.pop_back();
    if (!content || !stopwords().count(w)) out.push_back(w);
    w.clear();
  };
  for (char ch : text) {
    if (std::isalnum(static_cast<unsigned char>(ch))) w += static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    else flush();
  }
  flush();
  return out;
}

// Sentences: split after . ! ? followed by a space and an upper-case letter, a digit or a quote.
std::vector<std::string> sentences(const std::string& text) {
  std::vector<std::string> out;
  std::string cur;
  for (std::size_t i = 0; i < text.size(); ++i) {
    cur += text[i];
    const bool end = (text[i] == '.' || text[i] == '!' || text[i] == '?') && i + 2 < text.size() && text[i + 1] == ' ' &&
                     (std::isupper(static_cast<unsigned char>(text[i + 2])) || std::isdigit(static_cast<unsigned char>(text[i + 2])) || text[i + 2] == '"');
    if (end && words(cur, false).size() >= 4) {
      out.push_back(cur);
      cur.clear();
      ++i;  // the space
    }
  }
  if (!words(cur, false).empty()) {
    if (!out.empty() && words(cur, false).size() < 4) out.back() += cur;
    else out.push_back(cur);
  }
  for (auto& s : out) {
    s.erase(0, s.find_first_not_of(" \t\r\n"));
    s.erase(s.find_last_not_of(" \t\r\n") + 1);
  }
  return out;
}

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

const kernel::Model* readingQaModel() {
  static const kernel::Model model = [] {
    kernel::Model m;
    m.name = "readingQA";
    m.title = "Reading for answers: a reader learns word informativeness from a corpus and answers questions (QA golden test)";
    m.maxStats = 1;
    m.agentSource = reading_agent;
    m.compile = [](const Value& spec) {
      const Value& given = spec.sub("readingQA");
      const Value data = given.has("questions") ? given : example();
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
        qmeta.push(Value::object().set("passage", static_cast<double>(p)));
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
      if (given.has("questions") ? given.num("openBook", 1.0) > 0.5 : true) {
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
      Value answers = Value::array();
      for (std::uint32_t j = 0; j < plans[0].jobs; ++j) {
        std::size_t i = 0;
        double b = 0;
        choose(0, j, i, b);
        Value a = Value::object().set("reading", texts[i]).set("readingBelief", b).set("readingSentence", static_cast<double>(i));
        if (plans.size() > 1) {
          choose(1, j, i, b);
          a.set("openBook", texts[i]).set("openBookBelief", b).set("openBookSentence", static_cast<double>(i))
              .set("openBookDocument", docOf[i]);
        }
        answers.push(a);
      }
      return Value::object()
          .set("answers", answers)
          .set("sentences", meta.at("sentences"))
          .set("doc", meta.at("doc"))
          .set("docFirst", meta.at("docFirst"))
          .set("vocabulary", meta.at("vocabulary"))
          .set("documents", meta.at("documents"))
          .set("passages", meta.at("passages"));
    };
    return m;
  }();
  return &model;
}

}  // namespace syn::models
