// The end-to-end QA golden test: the readingQA model reads the test passages and the knowledge
// corpus, answers every golden question (on the software GPU, the same WGSL as WebGPU runs), and
// its answers are scored against the golden answers with SQuAD's token F1. Reference points: the
// passage's first sentence, a random sentence, and the oracle (the passage sentence closest to the
// golden answer: the best any one-sentence extractive reader can do).
//
// The reader also ingests every passage it reads as a knowledge capsule (capsuleIngest: each word
// a point of the space turned by the rotor of its grammatical role), recalls the capsule
// (capsuleRecall), and constructs answers from the recalled facts by its grammar — never from the
// text. As a student (readingStudent) it learns, from lessons on other passages (train.tsv), how to
// pick an answer, how to take notes and how to construct an answer, with every learning method of
// the library.
//
//   qa_golden DATA_DIR [--baseline FILE] [--write-baseline FILE] [--write-memory FILE]
//                      [--write-capsule PREFIX] [--capsule-passages N]
//
// DATA_DIR holds qa.tsv, corpus.tsv and train.tsv (tools/qa_golden/prepare.py). Writes
// DATA_DIR/report.md and DATA_DIR/answers.tsv. With --baseline, fails (exit 1) if a score drops
// more than 0.01 below the recorded one. With --write-memory, writes what the student learned in
// this run, starting from its memory, as its next memory (src/models/student_memory.inc); with
// --write-capsule, the capsule of the first N test passages (default 60, the embedded golden set)
// as PREFIX_lexicon.inc and PREFIX_points.inc (src/models/reader_capsule).
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
std::string memoryInc(const Value& habits, const Value& notes, const Value& construct, const Value& habitNames, const Value& noteNames,
                      const Value& constructNames, const std::string& fingerprint, const std::string& source) {
  std::ostringstream o;
  o.precision(17);
  auto weights = [&](const Value& next, const Value& names, const char* close) {
    const auto& w = next.at("weights");
    o << " {";
    for (std::size_t k = 0; k < 8; ++k)
      o << (k ? "  " : "") << w[k].asNumber() << (k < 7 ? "," : close) << "  // " << names.asArray()[k].at("habit").asString() << "\n";
  };
  o << "// The reader's memory: its reading, note-taking and constructing habits, learned from golden answers, kept between runs.\n"
    << "// Written by tools/qa_golden (qa_golden DATA_DIR --write-memory src/models/student_memory.inc). Do not edit by hand.\n"
    << "{" << habits.num("generation", 0) << ",  // generation\n " << habits.num("lessons", 0) << ",  // lessons studied, in all generations\n "
    << quoted(habits.at("design").asString()) << ",  // study design the reading habits were learned with\n " << quoted(source)
    << ",  // lessons it was learned from\n " << habits.num("test", 0) << ",  // test F1 of its answers on the golden test when it was written\n "
    << quoted(fingerprint) << ",  // the habits these weights are for\n";
  weights(habits, habitNames, "},");
  o << " " << quoted(notes.at("design").asString()) << ",  // study design the note-taking habits were learned with\n " << notes.num("test", 0)
    << ",  // test F1 of its best note on the golden test when it was written\n";
  weights(notes, noteNames, "},");
  o << " " << quoted(construct.at("design").asString()) << ",  // study design the constructing habits were learned with\n "
    << construct.num("test", 0) << ",  // test F1 of its constructed answers on the golden test when it was written\n";
  weights(construct, constructNames, "}}");
  return o.str();
}

// The capsule as C++ (src/models/reader_capsule_{lexicon,points}.inc): the lexicon, and a stream of
// numbers — the passages; for each, its vocabulary, its facts (slots in each) and its points.
void writeCapsule(const std::string& prefix, const Value& capsule, std::size_t passages) {
  std::ofstream lex(prefix + "_lexicon.inc");
  lex << "// The lexicon of the reader's knowledge capsule: the words it knows (no text). Written by tools/qa_golden\n"
      << "// (qa_golden DATA_DIR --write-capsule src/models/reader_capsule). Do not edit by hand.\n";
  const Value& dict = capsule.at("dictionary");
  for (std::size_t w = 0; w < dict.size(); ++w) lex << quoted(dict[w].asString()) << (w % 8 == 7 ? ",\n" : ", ");
  lex << "\n";
  std::ofstream pts(prefix + "_points.inc");
  pts << "// The reader's knowledge capsule: the " << passages << " golden passages it read, as points of the space — every word of a fact\n"
      << "// the point of S³ its coordinates lift to, turned by the rotor of its grammatical role (capsuleIngest). Per passage: its\n"
      << "// vocabulary (lexicon indices), its facts (slots in each), its points (5 coordinates per slot). Written by tools/qa_golden\n"
      << "// (qa_golden DATA_DIR --write-capsule src/models/reader_capsule). Do not edit by hand.\n"
      << passages << ",\n";
  pts.precision(7);
  const auto& ps = capsule.at("passages").asArray();
  for (std::size_t i = 0; i < passages && i < ps.size(); ++i) {
    for (const char* part : {"vocabulary", "facts"}) {
      const Value& v = ps[i].at(part);
      pts << v.size() << ",";
      for (std::size_t k = 0; k < v.size(); ++k) pts << v[k].asNumber() << ",";
      pts << "\n";
    }
    const Value& points = ps[i].at("points");
    for (std::size_t k = 0; k < points.size(); ++k) pts << points[k].asNumber() << (k % 5 == 4 ? ",\n" : ",");
  }
}

std::string field(const std::string& s) {
  std::string out = s;
  std::replace(out.begin(), out.end(), '\t', ' ');
  return out;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "usage: qa_golden DATA_DIR [--baseline FILE] [--write-baseline FILE] [--write-memory FILE] [--write-capsule PREFIX] "
                 "[--capsule-passages N]\n";
    return 2;
  }
  const std::string dir = argv[1];
  std::string baselineIn, baselineOut, memoryOut, capsuleOut;
  std::size_t capsulePassages = 60;
  for (int i = 2; i + 1 < argc; i += 2) {
    const std::string k = argv[i];
    if (k == "--baseline") baselineIn = argv[i + 1];
    else if (k == "--write-baseline") baselineOut = argv[i + 1];
    else if (k == "--write-memory") memoryOut = argv[i + 1];
    else if (k == "--write-capsule") capsuleOut = argv[i + 1];
    else if (k == "--capsule-passages") capsulePassages = static_cast<std::size_t>(std::max(1, std::stoi(argv[i + 1])));
  }
  const auto qa = readTsv(dir + "/qa.tsv");
  const auto corpus = readTsv(dir + "/corpus.tsv");
  const Value seed = Value::object().set("seed", 2026);
  auto run = [&](const std::string& model, const Value& body) {
    Value s = Value::object();
    s.set(model, body);
    s.set("sim", seed);
    return syn::kernel::run(model, s);
  };

  // The test: every test passage once, the questions (naming their passage and its capsule), and
  // the knowledge corpus. The lessons: golden questions about other passages.
  Value passages = Value::array(), questions = Value::array(), docs = Value::array(), test = Value::array(), testCapsules = Value::array();
  std::map<std::string, int> passageOf;
  for (const auto& r : qa) {
    if (r.size() < 4) continue;
    auto it = passageOf.find(r[0]);
    if (it == passageOf.end()) {
      it = passageOf.emplace(r[0], static_cast<int>(passageOf.size())).first;
      passages.push(r[3]);
    }
    questions.push(Value::object().set("question", r[1]).set("golden", r[2]).set("passage", it->second).set("capsule", it->second));
    test.push(Value::object().set("question", r[1]).set("golden", r[2]).set("passage", r[3]));
    testCapsules.push(Value::object().set("question", r[1]).set("golden", r[2]).set("capsule", it->second));
  }
  for (const auto& r : corpus)
    if (r.size() >= 2) docs.push(r[1]);
  Value train = Value::array(), trainCapsules = Value::array(), read = passages;
  for (const auto& r : readTsv(dir + "/train.tsv"))
    if (r.size() >= 4) {
      train.push(Value::object().set("question", r[1]).set("golden", r[2]).set("passage", r[3]));
      trainCapsules.push(Value::object().set("question", r[1]).set("golden", r[2]).set("capsule", static_cast<double>(read.size())));
      read.push(r[3]);
    }

  // The knowledge capsule: every passage it reads (the test's, then the lessons'), ingested into the
  // space and recalled from it.
  const Value ingested = run("capsuleIngest", Value::object().set("passages", read));
  const Value recall = run("capsuleRecall", Value::object().set("capsule", ingested.at("capsule")));
  const Value& recalled = recall.at("recalled");

  // The reader as a student: how to pick an answer, how to take notes, how to construct an answer.
  auto student = [&](bool notes, bool construct) {
    Value body = construct ? Value::object().set("train", trainCapsules).set("test", testCapsules).set("recalled", recalled).set("construct", true)
                           : Value::object().set("train", train).set("test", test).set("notes", notes);
    return run("readingStudent", body);
  };
  const Value st = student(false, false), notesSt = student(true, false), constructSt = student(false, true);
  const Value& nextHabits = st.at("remember");
  const Value& nextNotes = notesSt.at("remember");
  const Value& nextConstruct = constructSt.at("remember");

  // Answers: reading, open book and remembering from the text; constructed from the capsule alone.
  const Value result = run("readingQA", Value::object().set("passages", passages).set("questions", questions).set("corpus", docs).set("recalled", recalled));
  const auto& answers = result.at("answers").asArray();
  const Value& scores = result.at("scores");
  const Value& atLeastHalf = result.at("atLeastHalf");
  std::ofstream rows(dir + "/answers.tsv");
  rows << "question\tgolden\treader\tF1\topen book\tF1\toracle F1\tremembered\tF1\tconstructed\tF1\n";
  for (const auto& a : answers)
    rows << field(a.at("question").asString()) << '\t' << field(a.at("golden").asString()) << '\t' << field(a.at("reading").asString()) << '\t'
         << a.num("f1", 0) << '\t' << field(a.at("openBook").asString()) << '\t' << a.num("openBookF1", 0) << '\t' << a.num("oracle", 0) << '\t'
         << field(a.at("remembered").asString()) << '\t' << a.num("rememberedF1", 0) << '\t' << field(a.at("constructed").asString()) << '\t'
         << a.num("constructedF1", 0) << '\n';
  Value metrics = Value::object();
  std::ostringstream md;
  md << "# QA golden test\n\n" << answers.size() << " golden questions (edededdy/fineweb-edu-grounded-qa, FineWeb-Edu passages), "
     << result.num("passages", 0) << " passages + " << docs.size() << " knowledge documents read; vocabulary " << result.num("vocabulary", 0)
     << " words, " << result.at("sentences").size() << " sentences.\n\n| answer | token F1 | F1 ≥ 0.5 |\n|---|---:|---:|\n";
  const char* keys[7] = {"reading", "remembered", "constructed", "openBook", "firstSentence", "randomSentence", "oracle"};
  const char* names[7] = {"reader, its own passage (copies a sentence)", "reader with its remembered habits, its own passage (copies one or two sentences)",
                          "constructed from its knowledge capsule alone (never sees the text)", "reader, open book (the whole corpus)",
                          "first sentence of the passage", "a random sentence of the passage", "oracle: the best sentence of the passage"};
  for (int k = 0; k < 7; ++k) {
    md << "| " << names[k] << " | " << r3(scores.num(keys[k], 0)) << " | " << std::round(1000 * atLeastHalf.num(keys[k], 0)) / 10 << "% |\n";
    metrics.set(keys[k], scores.num(keys[k], 0));
  }
  const double found = scores.num("openBookFoundPassage", 0);
  md << "\nOpen book, the reader's answer came from the question's own passage for " << std::round(1000 * found) / 10 << "% of the questions.\n";
  metrics.set("openBookFoundPassage", found);

  // ---- The knowledge capsule.
  md << "\n## The knowledge capsule\n\nThe grammar read the " << read.size() << " passages (" << passages.size() << " of the test, " << train.size()
     << " of the lessons; " << ingested.num("wordsRead", 0) << " words) as " << ingested.num("facts", 0) << " facts — subject, relation, object — and "
     << "ingested them as " << ingested.num("points", 0) << " points of the space, each word turned by the grammar's rotors of its role and place (and kept out of order), over a lexicon of "
     << ingested.num("lexicon", 0) << " words. Recalled from the points alone, " << std::round(100000 * recall.num("recovered", 0)) / 1000
     << "% of them came back as their word, role and place. Constructed answers are said by the grammar from the recalled facts — the whole fact, "
     << "its predicate, its object or its subject — and chosen by the constructing habits; the text is never used.\n";
  metrics.set("capsuleRecovered", recall.num("recovered", 0));

  // ---- The reader as a student, with every learning method of the library.
  auto signed3 = [](double x) {
    std::ostringstream o;
    o << (x >= 0 ? "+" : "") << r3(x);
    return o.str();
  };
  auto section = [&](const Value& r, const std::string& title, const std::string& what) {
    const Value& mem = r.at("memory");
    md << "\n## " << title << "\n\n" << what << " " << r.num("lessons", 0) << " lessons (golden questions about other passages), three passes, "
       << r.num("tests", 0) << " test questions, " << r.num("students", 0) << " students per design; the best candidate scores " << r3(r.num("oracle", 0))
       << ". Its memory: "
       << (mem.at("used").asBool() ? "generation " + std::to_string(static_cast<int>(mem.num("generation", 0))) : mem.at("reason").asString()) << ", "
       << mem.num("lessons", 0) << " lessons studied before.\n\n| study design | test F1 | lessons F1 |\n|---|---:|---:|\n";
    for (const auto& d : r.at("designs").asArray())
      md << "| " << d.at("name").asString() << " | " << r3(d.num("test", 0)) << " | " << r3(d.num("lessons", 0)) << " |\n";
    md << "\n| method | library models | added to reinforcement | left out of all |\n|---|---|---:|---:|\n";
    for (const auto& m : r.at("methods").asArray())
      md << "| " << m.at("method").asString() << " | " << m.at("models").asString() << " | " << signed3(m.num("alone", 0)) << " | "
         << signed3(-m.num("removed", 0)) << " |\n";
  };
  section(st, "The reader as a student: picking an answer", "Candidates: one sentence of the passage or two adjacent ones.");
  section(constructSt, "The reader as a student: constructing an answer from its capsule",
          "Candidates: what the grammar can say from each fact recalled from the capsule — the whole fact, its predicate, its object, its subject.");
  section(notesSt, "The reader as a student: taking notes", "Candidates: the sentences of a passage, before any question.");
  auto named = [&](const Value& r, const std::string& name) {
    for (const auto& d : r.at("designs").asArray())
      if (d.at("name").asString() == name) return d.num("test", 0);
    return 0.0;
  };
  metrics.set("student", named(st, "reinforcement"));
  metrics.set("memory", named(st, "its memory alone, no feedback"));
  metrics.set("allMethods", named(st, "all methods"));
  metrics.set("constructingStudent", named(constructSt, "all methods"));

  const std::string source = std::to_string(train.size()) + " lessons of edededdy/fineweb-edu-grounded-qa (tools/qa_golden/prepare.py, seed 2026)";
  md << "\nIts next memory (generation " << nextHabits.num("generation", 0) << "): reading, note-taking and constructing habits learned with all "
     << "methods.\n";
  if (!memoryOut.empty()) {
    std::ofstream(memoryOut) << memoryInc(nextHabits, nextNotes, nextConstruct, st.at("designs").asArray()[0].at("habits"),
                                          notesSt.at("designs").asArray()[0].at("habits"), constructSt.at("designs").asArray()[0].at("habits"),
                                          st.at("fingerprint").asString(), source);
    md << "Memory written to " << memoryOut << ".\n";
  }
  if (!capsuleOut.empty()) {
    // The stored capsule: the first N test passages (the embedded golden set), ingested on their own.
    Value first = Value::array();
    for (std::size_t i = 0; i < capsulePassages && i < passages.size(); ++i) first.push(passages[i]);
    const Value own = run("capsuleIngest", Value::object().set("passages", first));
    writeCapsule(capsuleOut, own.at("capsule"), first.size());
    md << "Capsule of " << first.size() << " passages (" << own.num("points", 0) << " points, lexicon " << own.num("lexicon", 0) << ") written to "
       << capsuleOut << "_{lexicon,points}.inc.\n";
  }
  std::ofstream(dir + "/report.md") << md.str();
  std::cout << md.str();

  if (!baselineOut.empty()) std::ofstream(baselineOut) << metrics.json() << "\n";
  if (!baselineIn.empty()) {
    // The baseline is a flat JSON object of numbers: read "key": value pairs.
    std::ifstream in(baselineIn);
    std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    bool failed = false;
    for (const char* k : {"reading", "openBook", "openBookFoundPassage", "student", "memory", "allMethods", "remembered", "constructed",
                          "constructingStudent", "capsuleRecovered"}) {
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
