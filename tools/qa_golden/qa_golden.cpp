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
    questions.push(Value::object().set("question", r[1]).set("golden", r[2]).set("passage", it->second));
  }
  for (const auto& r : corpus)
    if (r.size() >= 2) docs.push(r[1]);
  Value spec = Value::object();
  spec.set("readingQA", Value::object().set("passages", passages).set("questions", questions).set("corpus", docs));
  spec.set("sim", Value::object().set("seed", 2026));

  const Value result = syn::kernel::run("readingQA", spec);
  const auto& answers = result.at("answers").asArray();
  const Value& scores = result.at("scores");
  const Value& atLeastHalf = result.at("atLeastHalf");
  std::ofstream rows(dir + "/answers.tsv");
  rows << "question\tgolden\treader\tF1\topen book\tF1\toracle F1\n";
  for (const auto& a : answers)
    rows << field(a.at("question").asString()) << '\t' << field(a.at("golden").asString()) << '\t' << field(a.at("reading").asString()) << '\t'
         << a.num("f1", 0) << '\t' << field(a.at("openBook").asString()) << '\t' << a.num("openBookF1", 0) << '\t' << a.num("oracle", 0) << '\n';
  Value metrics = Value::object();
  std::ostringstream md;
  md << "# QA golden test\n\n" << answers.size() << " golden questions (edededdy/fineweb-edu-grounded-qa, FineWeb-Edu passages), "
     << result.num("passages", 0) << " passages + " << docs.size() << " knowledge documents read; vocabulary "
     << result.num("vocabulary", 0) << " words, " << result.at("sentences").size() << " sentences.\n\n| answer | token F1 | F1 ≥ 0.5 |\n|---|---:|---:|\n";
  const char* keys[5] = {"reading", "openBook", "firstSentence", "randomSentence", "oracle"};
  const char* names[5] = {"reader, its own passage", "reader, open book (the whole corpus)", "first sentence of the passage",
                          "a random sentence of the passage", "oracle: the best sentence of the passage"};
  for (int k = 0; k < 5; ++k) {
    md << "| " << names[k] << " | " << std::round(1000 * scores.num(keys[k], 0)) / 1000 << " | " << std::round(1000 * atLeastHalf.num(keys[k], 0)) / 10 << "% |\n";
    metrics.set(keys[k], scores.num(keys[k], 0));
  }
  const double found = scores.num("openBookFoundPassage", 0);
  md << "\nOpen book: the reader's answer came from the question's own passage for " << std::round(1000 * found) / 10 << "% of the questions.\n";
  metrics.set("openBookFoundPassage", found);
  // The reader as a student: lessons from train.tsv (other passages), the same test questions.
  if (std::ifstream(dir + "/train.tsv")) {
    Value train = Value::array(), test = Value::array();
    for (const auto& r : readTsv(dir + "/train.tsv"))
      if (r.size() >= 4) train.push(Value::object().set("question", r[1]).set("golden", r[2]).set("passage", r[3]));
    for (const auto& r : qa)
      if (r.size() >= 4) test.push(Value::object().set("question", r[1]).set("golden", r[2]).set("passage", r[3]));
    Value sspec = Value::object();
    sspec.set("readingStudent", Value::object().set("train", train).set("test", test));
    sspec.set("sim", Value::object().set("seed", 2026));
    const Value st = syn::kernel::run("readingStudent", sspec);
    md << "\n## The reader as a student\n\n" << st.num("lessons", 0) << " lessons (golden questions about other passages), " << st.num("tests", 0)
       << " test questions, " << st.num("students", 0) << " students per design. Candidates: one sentence or two adjacent ones; the best of them "
       << "scores " << std::round(1000 * st.num("oracle", 0)) / 1000 << ".\n\n| study design | test F1 | lessons F1 | two-sentence answers |\n|---|---:|---:|---:|\n";
    for (const auto& d : st.at("designs").asArray()) {
      md << "| " << d.at("name").asString() << " | " << std::round(1000 * d.num("test", 0)) / 1000 << " | " << std::round(1000 * d.num("lessons", 0)) / 1000
         << " | " << std::round(1000 * d.num("twoSentences", 0)) / 10 << "% |\n";
    }
    metrics.set("student", st.at("designs").asArray()[1].num("test", 0));
  }
  std::ofstream(dir + "/report.md") << md.str();
  std::cout << md.str();

  if (!baselineOut.empty()) std::ofstream(baselineOut) << metrics.json() << "\n";
  if (!baselineIn.empty()) {
    // The baseline is a flat JSON object of numbers: read "key": value pairs.
    std::ifstream in(baselineIn);
    std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    bool failed = false;
    for (const char* k : {"reading", "openBook", "openBookFoundPassage", "student"}) {
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
