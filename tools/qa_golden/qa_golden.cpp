// The end-to-end QA golden test: the readingQA model reads the test passages and the knowledge
// corpus, answers every golden question (on the software GPU, the same WGSL as WebGPU runs), and
// its answers are scored against the golden answers with SQuAD's token F1. Reference points: the
// passage's first sentence, a random sentence, and the oracle (the passage sentence closest to the
// golden answer: the best any one-sentence extractive reader can do).
//
//   qa_golden DATA_DIR [--baseline FILE] [--write-baseline FILE]
//
// DATA_DIR holds qa.tsv and corpus.tsv (tools/qa_golden/prepare.py). Writes DATA_DIR/report.md
// and DATA_DIR/answers.tsv. With --baseline, fails (exit 1) if a score drops more than 0.01 below
// the recorded one.
#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "syn/core/value.hpp"
#include "syn/kernel/kernel.hpp"

using syn::Value;

namespace {

std::vector<std::vector<std::string>> readTsv(const std::string& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("cannot read " + path);
  std::vector<std::vector<std::string>> rows;
  std::string line;
  while (std::getline(in, line)) {
    std::vector<std::string> cols;
    std::stringstream ss(line);
    std::string c;
    while (std::getline(ss, c, '\t')) cols.push_back(c);
    if (!cols.empty()) rows.push_back(cols);
  }
  return rows;
}

// SQuAD normalisation: lower case, no punctuation, no articles, single spaces.
std::vector<std::string> normalised(const std::string& s) {
  std::vector<std::string> out;
  std::string w;
  auto flush = [&]() {
    if (!w.empty() && w != "a" && w != "an" && w != "the") out.push_back(w);
    w.clear();
  };
  for (char ch : s) {
    if (std::isalnum(static_cast<unsigned char>(ch))) w += static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    else if (std::isspace(static_cast<unsigned char>(ch))) flush();
  }
  flush();
  return out;
}

double f1(const std::string& prediction, const std::string& golden) {
  const auto p = normalised(prediction), g = normalised(golden);
  if (p.empty() || g.empty()) return p == g ? 1.0 : 0.0;
  std::map<std::string, int> count;
  for (const auto& w : g) count[w]++;
  int common = 0;
  for (const auto& w : p)
    if (count[w]-- > 0) ++common;
  if (common == 0) return 0.0;
  const double precision = double(common) / double(p.size()), recall = double(common) / double(g.size());
  return 2 * precision * recall / (precision + recall);
}

std::string field(const std::string& s) {
  std::string out = s;
  std::replace(out.begin(), out.end(), '\t', ' ');
  return out;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "usage: qa_golden DATA_DIR [--baseline FILE] [--write-baseline FILE]\n";
    return 2;
  }
  const std::string dir = argv[1];
  std::string baselineIn, baselineOut;
  for (int i = 2; i + 1 < argc; i += 2) {
    const std::string k = argv[i];
    if (k == "--baseline") baselineIn = argv[i + 1];
    else if (k == "--write-baseline") baselineOut = argv[i + 1];
  }
  const auto qa = readTsv(dir + "/qa.tsv");
  const auto corpus = readTsv(dir + "/corpus.tsv");

  // The spec: every test passage once, the questions, and the knowledge corpus.
  Value passages = Value::array(), questions = Value::array(), docs = Value::array();
  std::map<std::string, int> passageOf;
  for (const auto& r : qa) {
    if (r.size() < 4) continue;
    auto it = passageOf.find(r[0]);
    if (it == passageOf.end()) {
      it = passageOf.emplace(r[0], static_cast<int>(passageOf.size())).first;
      passages.push(r[3]);
    }
    questions.push(Value::object().set("question", r[1]).set("passage", it->second));
  }
  for (const auto& r : corpus)
    if (r.size() >= 2) docs.push(r[1]);
  Value spec = Value::object();
  spec.set("readingQA", Value::object().set("passages", passages).set("questions", questions).set("corpus", docs));
  spec.set("sim", Value::object().set("seed", 2026));

  const Value result = syn::kernel::run("readingQA", spec);
  const auto& answers = result.at("answers").asArray();
  const auto& sentences = result.at("sentences").asArray();
  const auto& docFirst = result.at("docFirst").asArray();
  const auto nSentences = sentences.size();

  struct Score {
    const char* name;
    double f1 = 0, half = 0;
  };
  std::vector<Score> scores = {{"reader, its own passage"}, {"reader, open book (the whole corpus)"}, {"first sentence of the passage"},
                               {"a random sentence of the passage"}, {"oracle: the best sentence of the passage"}};
  double found = 0;
  std::ofstream rows(dir + "/answers.tsv");
  rows << "question\tgolden\treader\tF1\topen book\tF1\toracle F1\n";
  std::uint32_t rng = 2026;
  for (std::size_t q = 0; q < answers.size(); ++q) {
    const auto& golden = qa[q][2];
    const auto p = static_cast<std::size_t>(questions.asArray()[q].at("passage").asNumber());
    const auto first = static_cast<std::size_t>(docFirst[p].asNumber());
    const auto last = p + 1 < docFirst.size() ? static_cast<std::size_t>(docFirst[p + 1].asNumber()) : nSentences;
    rng = rng * 1664525u + 1013904223u;
    const std::size_t randomIndex = first + (last > first ? (rng >> 8) % (last - first) : 0);
    double oracle = 0;
    for (std::size_t s = first; s < last; ++s) oracle = std::max(oracle, f1(sentences[s].asString(), golden));
    const double f[5] = {f1(answers[q].at("reading").asString(), golden), f1(answers[q].at("openBook").asString(), golden),
                         first < last ? f1(sentences[first].asString(), golden) : 0.0,
                         first < last ? f1(sentences[randomIndex].asString(), golden) : 0.0, oracle};
    for (int k = 0; k < 5; ++k) {
      scores[k].f1 += f[k];
      scores[k].half += f[k] >= 0.5 ? 1.0 : 0.0;
    }
    if (static_cast<std::size_t>(answers[q].at("openBookDocument").asNumber()) == p) found += 1;
    rows << field(qa[q][1]) << '\t' << field(golden) << '\t' << field(answers[q].at("reading").asString()) << '\t' << f[0] << '\t'
         << field(answers[q].at("openBook").asString()) << '\t' << f[1] << '\t' << oracle << '\n';
  }
  const double n = static_cast<double>(std::max<std::size_t>(1, answers.size()));
  Value metrics = Value::object();
  std::ostringstream md;
  md << "# QA golden test\n\n" << answers.size() << " golden questions (edededdy/fineweb-edu-grounded-qa, FineWeb-Edu passages), "
     << result.num("passages", 0) << " passages + " << docs.size() << " knowledge documents read; vocabulary "
     << result.num("vocabulary", 0) << " words, " << nSentences << " sentences.\n\n| answer | token F1 | F1 ≥ 0.5 |\n|---|---:|---:|\n";
  const char* keys[5] = {"reading", "openBook", "firstSentence", "randomSentence", "oracle"};
  for (int k = 0; k < 5; ++k) {
    md << "| " << scores[k].name << " | " << std::round(1000 * scores[k].f1 / n) / 1000 << " | " << std::round(1000 * scores[k].half / n) / 10 << "% |\n";
    metrics.set(keys[k], scores[k].f1 / n);
  }
  md << "\nOpen book: the reader's answer came from the question's own passage for " << std::round(1000 * found / n) / 10 << "% of the questions.\n";
  metrics.set("openBookFoundPassage", found / n);
  std::ofstream(dir + "/report.md") << md.str();
  std::cout << md.str();

  if (!baselineOut.empty()) std::ofstream(baselineOut) << metrics.json() << "\n";
  if (!baselineIn.empty()) {
    // The baseline is a flat JSON object of numbers: read "key": value pairs.
    std::ifstream in(baselineIn);
    std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    bool failed = false;
    for (const char* k : {"reading", "openBook", "openBookFoundPassage"}) {
      const auto at = text.find(std::string("\"") + k + "\":");
      if (at == std::string::npos) continue;
      const double recorded = std::stod(text.substr(at + std::string(k).size() + 3));
      const double now = metrics.num(k, 0);
      if (now < recorded - 0.01) {
        std::cout << "REGRESSION " << k << ": " << now << " < " << recorded << "\n";
        failed = true;
      }
    }
    std::cout << (failed ? "golden test FAILED\n" : "golden test passed (no score below its baseline)\n");
    return failed ? 1 : 0;
  }
  return 0;
}
