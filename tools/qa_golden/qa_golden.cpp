// The end-to-end QA golden test: the readingQA model reads the test passages and the knowledge
// corpus, answers every golden question (on the software GPU, the same WGSL as WebGPU runs), and
// its answers are scored against the golden answers with SQuAD's token F1. Reference points: the
// passage's first sentence, a random sentence, and the oracle (the passage sentence closest to the
// golden answer: the best any one-sentence extractive reader can do).
//
// The reader also learns, as a student (readingStudent), from lessons on other passages (train.tsv):
// how to pick an answer, and how to take notes. It takes notes on the test passages (readingNotes)
// and answers from its notes alone, as if the passages were gone (closed book), and with the reading
// habits it remembers.
//
//   qa_golden DATA_DIR [--baseline FILE] [--write-baseline FILE] [--write-memory FILE]
//                      [--write-knowledge FILE] [--keep N]
//
// DATA_DIR holds qa.tsv, corpus.tsv and train.tsv (tools/qa_golden/prepare.py). Writes
// DATA_DIR/report.md, DATA_DIR/answers.tsv and DATA_DIR/knowledge.tsv (the notes). With --baseline,
// fails (exit 1) if a score drops more than 0.01 below the recorded one. With --write-memory, writes
// what the student learned in this run, starting from its memory, as its next memory
// (src/models/student_memory.inc); with --write-knowledge, the notes it took with that memory
// (src/models/reader_knowledge.inc), --keep sentences per passage (default 3).
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

double r3(double x) { return std::round(1000 * x) / 1000; }

std::string quoted(const std::string& s) {
  std::string out = "\"";
  for (char c : s) {
    if (c == '"' || c == '\\') out += '\\';
    out += c;
  }
  return out + "\"";
}

// The memory as C++ (src/models/student_memory.inc): what the next run of the reader starts from.
std::string memoryInc(const Value& habits, const Value& notes, const Value& habitNames, const Value& noteNames, const std::string& fingerprint,
                      const std::string& source) {
  std::ostringstream o;
  o.precision(17);
  auto weights = [&](const Value& next, const Value& names, const char* close) {
    const auto& w = next.at("weights");
    o << " {";
    for (std::size_t k = 0; k < 8; ++k)
      o << (k ? "  " : "") << w[k].asNumber() << (k < 7 ? "," : close) << "  // " << names[k].at("habit").asString() << "\n";
  };
  o << "// The reader's memory: its reading and note-taking habits, learned from golden answers, kept between runs.\n"
    << "// Written by tools/qa_golden (qa_golden DATA_DIR --write-memory src/models/student_memory.inc). Do not edit by hand.\n"
    << "{" << habits.num("generation", 0) << ",  // generation\n " << habits.num("lessons", 0) << ",  // lessons studied, in all generations\n "
    << quoted(habits.at("design").asString()) << ",  // study design the reading habits were learned with\n " << quoted(source)
    << ",  // lessons it was learned from\n " << habits.num("test", 0) << ",  // test F1 of its answers on the golden test when it was written\n "
    << quoted(fingerprint) << ",  // the habits these weights are for\n";
  weights(habits, habitNames, "},");
  o << " " << quoted(notes.at("design").asString()) << ",  // study design the note-taking habits were learned with\n " << notes.num("test", 0)
    << ",  // test F1 of its best note on the golden test when it was written\n";
  weights(notes, noteNames, "}}");
  return o.str();
}

// The knowledge as C++ (src/models/reader_knowledge.inc): the notes on each passage read.
std::string knowledgeInc(const std::vector<std::string>& ids, const std::vector<std::string>& passages, const Value& notes, int generation, int keep) {
  std::ostringstream o;
  o << "// The reader's knowledge: the notes it took on the passages of the QA golden test (" << keep << " sentences each, chosen by the\n"
    << "// note-taking habits of memory generation " << generation << ", before any question), so that it can answer when the passages are gone.\n"
    << "// Written by tools/qa_golden (qa_golden DATA_DIR --write-knowledge src/models/reader_knowledge.inc). Do not edit by hand.\n"
    << "// Entries: passage (doc_index of edededdy/fineweb-edu-grounded-qa), memory generation, sentences kept, the passage's length, the notes.\n";
  for (std::size_t i = 0; i < ids.size(); ++i) {
    const std::string n = notes.at("notes")[i].asString();
    if (n.find(")kn\"") != std::string::npos) throw std::runtime_error("a note holds the raw-string delimiter");
    o << "{" << quoted(ids[i]) << ", " << generation << ", " << keep << ", " << passages[i].size() << ", R\"kn(" << n << ")kn\"},\n";
  }
  return o.str();
}

std::string field(const std::string& s) {
  std::string out = s;
  std::replace(out.begin(), out.end(), '\t', ' ');
  return out;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "usage: qa_golden DATA_DIR [--baseline FILE] [--write-baseline FILE] [--write-memory FILE] [--write-knowledge FILE] [--keep N]\n";
    return 2;
  }
  const std::string dir = argv[1];
  std::string baselineIn, baselineOut, memoryOut, knowledgeOut;
  int keep = 3;
  for (int i = 2; i + 1 < argc; i += 2) {
    const std::string k = argv[i];
    if (k == "--baseline") baselineIn = argv[i + 1];
    else if (k == "--write-baseline") baselineOut = argv[i + 1];
    else if (k == "--write-memory") memoryOut = argv[i + 1];
    else if (k == "--write-knowledge") knowledgeOut = argv[i + 1];
    else if (k == "--keep") keep = std::max(1, std::stoi(argv[i + 1]));
  }
  const auto qa = readTsv(dir + "/qa.tsv");
  const auto corpus = readTsv(dir + "/corpus.tsv");
  const Value seed = Value::object().set("seed", 2026);

  // The test: every test passage once, the questions, and the knowledge corpus.
  Value passages = Value::array(), questions = Value::array(), docs = Value::array(), test = Value::array();
  std::vector<std::string> ids, passageTexts;
  std::map<std::string, int> passageOf;
  for (const auto& r : qa) {
    if (r.size() < 4) continue;
    auto it = passageOf.find(r[0]);
    if (it == passageOf.end()) {
      it = passageOf.emplace(r[0], static_cast<int>(passageOf.size())).first;
      passages.push(r[3]);
      ids.push_back(r[0]);
      passageTexts.push_back(r[3]);
    }
    questions.push(Value::object().set("question", r[1]).set("golden", r[2]).set("passage", it->second));
    test.push(Value::object().set("question", r[1]).set("golden", r[2]).set("passage", r[3]));
  }
  for (const auto& r : corpus)
    if (r.size() >= 2) docs.push(r[1]);
  Value goldenOfPassage = Value::array();
  for (const auto& q : questions.asArray()) goldenOfPassage.push(q.at("golden"));
  Value train = Value::array();
  for (const auto& r : readTsv(dir + "/train.tsv"))
    if (r.size() >= 4) train.push(Value::object().set("question", r[1]).set("golden", r[2]).set("passage", r[3]));

  // The reader as a student: lessons on other passages, the same test questions; reading habits
  // and note-taking habits, fresh and from its memory.
  auto student = [&](bool memory, bool notes) {
    Value s = Value::object();
    s.set("readingStudent", Value::object().set("train", train).set("test", test).set("memory", memory).set("notes", notes));
    s.set("sim", seed);
    return syn::kernel::run("readingStudent", s);
  };
  const Value fresh = student(false, false), st = student(true, false), notesFresh = student(false, true), notesSt = student(true, true);
  const Value& nextHabits = st.at("remember");
  const Value& nextNotes = notesSt.at("remember");

  // Notes on the test passages with the note-taking habits it now has, before any question; the
  // knowledge is the notes of --keep sentences per passage.
  auto takeNotes = [&](int k) {
    Value s = Value::object();
    s.set("readingNotes", Value::object().set("passages", passages).set("golden", goldenOfPassage).set("keep", k).set("weights", nextNotes.at("weights")));
    s.set("sim", seed);
    return syn::kernel::run("readingNotes", s);
  };
  const Value notes = takeNotes(keep);
  {
    std::ofstream kn(dir + "/knowledge.tsv");
    for (std::size_t i = 0; i < ids.size(); ++i) kn << ids[i] << '\t' << notes.at("notes")[i].asString() << '\n';
  }
  auto ask = [&](const Value& knowledge, bool all) {
    Value s = Value::object();
    Value q = Value::object().set("passages", passages).set("questions", questions).set("corpus", docs).set("knowledge", knowledge);
    if (!all) q.set("openBook", 0).set("remember", 0);
    s.set("readingQA", q);
    s.set("sim", seed);
    return syn::kernel::run("readingQA", s);
  };
  const Value result = ask(notes.at("notes"), true);

  const auto& answers = result.at("answers").asArray();
  const Value& scores = result.at("scores");
  const Value& atLeastHalf = result.at("atLeastHalf");
  std::ofstream rows(dir + "/answers.tsv");
  rows << "question\tgolden\treader\tF1\topen book\tF1\toracle F1\tremembered\tF1\tknowledge\tF1\n";
  for (const auto& a : answers)
    rows << field(a.at("question").asString()) << '\t' << field(a.at("golden").asString()) << '\t' << field(a.at("reading").asString()) << '\t'
         << a.num("f1", 0) << '\t' << field(a.at("openBook").asString()) << '\t' << a.num("openBookF1", 0) << '\t' << a.num("oracle", 0) << '\t'
         << field(a.at("remembered").asString()) << '\t' << a.num("rememberedF1", 0) << '\t' << field(a.at("knowledge").asString()) << '\t'
         << a.num("knowledgeF1", 0) << '\n';
  Value metrics = Value::object();
  std::ostringstream md;
  md << "# QA golden test\n\n" << answers.size() << " golden questions (edededdy/fineweb-edu-grounded-qa, FineWeb-Edu passages), "
     << result.num("passages", 0) << " passages + " << docs.size() << " knowledge documents read; vocabulary " << result.num("vocabulary", 0)
     << " words, " << result.at("sentences").size() << " sentences.\n\n| answer | token F1 | F1 ≥ 0.5 |\n|---|---:|---:|\n";
  const char* keys[7] = {"reading", "remembered", "openBook", "knowledge", "firstSentence", "randomSentence", "oracle"};
  const char* names[7] = {"reader, its own passage", "reader with its remembered habits, its own passage", "reader, open book (the whole corpus)",
                          "reader from its knowledge alone (its notes; the passages gone)", "first sentence of the passage",
                          "a random sentence of the passage", "oracle: the best sentence of the passage"};
  for (int k = 0; k < 7; ++k) {
    md << "| " << names[k] << " | " << r3(scores.num(keys[k], 0)) << " | " << std::round(1000 * atLeastHalf.num(keys[k], 0)) / 10 << "% |\n";
    metrics.set(keys[k], scores.num(keys[k], 0));
  }
  const double found = scores.num("openBookFoundPassage", 0), knownFound = scores.num("knowledgeFoundPassage", 0);
  md << "\nOpen book, the reader's answer came from the question's own passage for " << std::round(1000 * found) / 10
     << "% of the questions; from its knowledge, its answer came from the notes on the question's passage for " << std::round(1000 * knownFound) / 10
     << "%.\n";
  metrics.set("openBookFoundPassage", found);

  // ---- The reader as a student.
  const Value& mem = st.at("memory");
  md << "\n## The reader as a student\n\n" << st.num("lessons", 0) << " lessons (golden questions about other passages), " << st.num("tests", 0)
     << " test questions, " << st.num("students", 0) << " students per design. Candidates: one sentence or two adjacent ones; the best of them "
     << "scores " << r3(st.num("oracle", 0)) << ".\n\n";
  md << "Its memory: " << (mem.at("used").asBool() ? "generation " + std::to_string(static_cast<int>(mem.num("generation", 0))) : mem.at("reason").asString())
     << ", " << mem.num("lessons", 0) << " lessons studied before" << (mem.at("design").asString().empty() ? "" : " (" + mem.at("design").asString() + ")")
     << ".\n\n| study design | test F1, fresh | test F1, from memory | lessons F1, from memory | two-sentence answers |\n|---|---:|---:|---:|---:|\n";
  const auto& fd = fresh.at("designs").asArray();
  const auto& sd = st.at("designs").asArray();
  for (std::size_t j = 0; j < sd.size(); ++j)
    md << "| " << sd[j].at("name").asString() << " | " << r3(fd[j].num("test", 0)) << " | " << r3(sd[j].num("test", 0)) << " | "
       << r3(sd[j].num("lessons", 0)) << " | " << std::round(1000 * sd[j].num("twoSentences", 0)) / 10 << "% |\n";
  metrics.set("student", fd[1].num("test", 0));
  metrics.set("memory", sd[0].num("test", 0));

  // ---- Taking notes.
  md << "\n## Taking notes, and answering from them\n\nThe student also learns which sentences of a passage are worth keeping before any question "
     << "is asked: the lesson's golden answer says which one was. Test F1 here is the F1 of the single sentence it would keep first.\n\n"
     << "| study design | best note F1, fresh | best note F1, from memory |\n|---|---:|---:|\n";
  const auto& nfd = notesFresh.at("designs").asArray();
  const auto& nsd = notesSt.at("designs").asArray();
  for (std::size_t j = 0; j < nsd.size(); ++j)
    md << "| " << nsd[j].at("name").asString() << " | " << r3(nfd[j].num("test", 0)) << " | " << r3(nsd[j].num("test", 0)) << " |\n";
  md << "\nWith the note-taking habits of \"" << nextNotes.at("design").asString() << "\" (the design that did best on its own lessons), it takes "
     << "notes on the " << ids.size() << " test passages and answers the questions from the notes alone:\n\n"
     << "| sentences kept per passage | text kept | best kept sentence F1 | answers from the notes, F1 |\n|---:|---:|---:|---:|\n";
  for (int k : {1, 2, 3, 5, 100}) {
    const Value n = k == keep ? notes : takeNotes(k);
    const Value r = ask(n.at("notes"), false);
    md << "| " << (k == 100 ? std::string("all") : std::to_string(k)) << " | " << std::round(1000 * n.num("kept", 0)) / 10 << "% | "
       << r3(n.num("bestKeptF1", 0)) << " | " << r3(r.at("scores").num("knowledge", 0)) << " |\n";
  }
  metrics.set("knowledgeKept", notes.num("kept", 0));

  const std::string source = std::to_string(train.size()) + " lessons of edededdy/fineweb-edu-grounded-qa (tools/qa_golden/prepare.py, seed 2026)";
  md << "\nIts next memory (generation " << nextHabits.num("generation", 0) << "): reading habits of \"" << nextHabits.at("design").asString()
     << "\", note-taking habits of \"" << nextNotes.at("design").asString() << "\".\n";
  if (!memoryOut.empty()) {
    std::ofstream(memoryOut) << memoryInc(nextHabits, nextNotes, sd[0].at("habits"), nsd[0].at("habits"), st.at("fingerprint").asString(), source);
    md << "Memory written to " << memoryOut << ".\n";
  }
  if (!knowledgeOut.empty()) {
    std::ofstream(knowledgeOut) << knowledgeInc(ids, passageTexts, notes, static_cast<int>(nextHabits.num("generation", 0)), keep);
    md << "Knowledge (" << keep << " sentences per passage, " << std::round(1000 * notes.num("kept", 0)) / 10 << "% of the text) written to "
       << knowledgeOut << ".\n";
  }
  std::ofstream(dir + "/report.md") << md.str();
  std::cout << md.str();

  if (!baselineOut.empty()) std::ofstream(baselineOut) << metrics.json() << "\n";
  if (!baselineIn.empty()) {
    // The baseline is a flat JSON object of numbers: read "key": value pairs.
    std::ifstream in(baselineIn);
    std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    bool failed = false;
    for (const char* k : {"reading", "openBook", "openBookFoundPassage", "student", "memory", "remembered", "knowledge"}) {
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
