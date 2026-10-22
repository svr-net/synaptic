// The reader as a native tool (reader.hpp).
#include "reader.hpp"

#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>

#include "../../src/models/reading_memory.hpp"
#include "syn/core/value.hpp"
#include "syn/kernel/kernel.hpp"

namespace syn::reader {

namespace memory = syn::models::memory;

namespace {

std::string readFile(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("cannot read " + path);
  std::ostringstream s;
  s << in.rdbuf();
  return s.str();
}

Value list(const std::vector<std::string>& xs) {
  Value v = Value::array();
  for (const auto& x : xs) v.push(x);
  return v;
}

// What it knows -- its stored capsule (the golden passages) -- and what it remembers, decoded once.
struct Knowledge {
  Value capsule;
  Value records;
};

Knowledge know(const std::vector<std::string>& records) {
  return {memory::capsuleOf(Value::object()).at("capsule"), list(records)};
}

Value info(const Knowledge& k) {
  double facts = 0;
  for (const auto& p : k.capsule.at("passages").asArray()) facts += static_cast<double>(p.at("facts").size());
  const memory::Memory mem = memory::recallFor(Value::object().set("memoryRecords", k.records));
  return Value::object()
      .set("passages", static_cast<double>(k.capsule.at("passages").size()))
      .set("facts", facts)
      .set("words", static_cast<double>(k.capsule.at("dictionary").size()))
      .set("memory", Value::object().set("generation", mem.generation).set("lessons", mem.lessons).set("records", mem.records));
}

// capsuleQA over what it knows: one run for every question.
Value answer(const Knowledge& k, const std::vector<std::string>& questions) {
  Value qs = Value::array();
  for (const auto& q : questions) qs.push(Value::object().set("question", q));
  // asked what something is, it prefers the fact that says what something is
  Value c = Value::object().set("capsule", k.capsule).set("questions", qs).set("definitionWeight", 1.0);
  if (k.records.size() > 0) c.set("memoryRecords", k.records);
  Value spec = Value::object();
  spec.set("capsuleQA", c);
  return kernel::run("capsuleQA", spec);
}

// Found: its evidence holds some of what the question asks. Confident: every part's evidence covers at least
// half of what the part asks -- and, where the part asks what something is ("What is X?"), its fact says what X is
// (X its subject, its relation a form of "be", "means", "is called"), not merely mentions X. Why it is not, when it is
// found but not confident: a part that asks what something is answered by a fact that does not say what it is, or
// evidence covering less than half of what the question asks.
std::string line(const Value& a) {
  Value v = a;
  const bool said = !a.at("answer").asString().empty();
  bool confident = said, found = false, undefined = false;
  for (const auto& p : a.at("parts").asArray()) {
    confident = confident && p.num("covers", 0) >= 0.5;
    const bool asks = p.has("definition") && p.at("definition").asBool(), says = p.has("defines") && p.at("defines").asBool();
    if (asks) confident = confident && says;
    if (asks && !says && p.has("answer") && !p.at("answer").asString().empty()) undefined = true;
    found = found || p.num("covers", 0) > 0;
  }
  found = said && found, confident = found && confident;
  v.set("found", found).set("confident", confident)
      .set("why", !found || confident ? ""
                  : undefined         ? "the fact it answers with does not say what that is"
                                      : "its evidence covers less than half of what the question asks");
  return v.json();
}

}  // namespace

std::string answerFrom(const std::vector<std::string>& passages, const std::string& question, const std::vector<std::string>& records) {
  std::vector<std::string> kept;
  for (const auto& p : passages)
    if (p.find_first_not_of(" \t\r\n") != std::string::npos) kept.push_back(p);
  if (kept.empty()) return Value::object().set("found", false).set("confident", false).set("answer", "").set("parts", Value::array())
                           .set("passages", 0.0).json();
  Value spec = Value::object();
  spec.set("capsuleIngest", Value::object().set("passages", list(kept)));
  const Value ing = kernel::run("capsuleIngest", spec);
  const Knowledge k{ing.at("capsule"), list(records)};
  Value a = Value::object();
  {
    const Value answered = answer(k, {question});
    a = answered.at("answers").asArray()[0];
  }
  // the line, with how many passages it read
  const std::string l = line(a);
  return l.substr(0, l.size() - 1) + ",\"passages\":" + std::to_string(kept.size()) + "}";
}

// How much worse than before the golden questions may be answered with a study kept (tools/qa_golden's baseline
// tolerance).
constexpr double kGoldenDrop = 0.01;

std::string practise(const std::vector<Lesson>& lessons, const std::vector<std::string>& records, const std::string& source) {
  std::vector<const Lesson*> usable;
  for (const auto& l : lessons)
    if (!l.question.empty() && !l.golden.empty() && !l.passages.empty()) usable.push_back(&l);
  Value out = Value::object().set("practised", static_cast<double>(usable.size())).set("kept", false);
  if (usable.size() < 8) return out.set("reason", "fewer than 8 lessons: nothing to learn from yet").json();
  // Each lesson's passages, one capsule each: read into the space and recalled from the codes, as it answers.
  Value passages = Value::array();
  for (const auto* l : usable) {
    std::string joined;
    for (const auto& p : l->passages) joined += (joined.empty() ? "" : " ") + p;
    passages.push(joined);
  }
  Value spec = Value::object();
  spec.set("capsuleIngest", Value::object().set("passages", passages));
  const Value ing = kernel::run("capsuleIngest", spec);
  Value rspec = Value::object();
  rspec.set("capsuleRecall", Value::object().set("capsule", ing.at("capsule")));
  const Value recalled = kernel::run("capsuleRecall", rspec).at("recalled");
  // Three quarters to study, every fourth held out.
  Value train = Value::array(), test = Value::array();
  for (std::size_t i = 0; i < usable.size(); ++i)
    (i % 4 == 3 ? test : train).push(Value::object().set("question", usable[i]->question).set("golden", usable[i]->golden).set("capsule", static_cast<double>(i)));
  auto study = [&](const std::vector<std::string>& remembered) {
    Value c = Value::object().set("construct", true).set("train", train).set("test", test).set("recalled", recalled).set("students", 16.0)
                  .set("source", source);
    if (!remembered.empty()) c.set("memoryRecords", list(remembered));
    Value s2 = Value::object();
    s2.set("readingStudent", c);
    return kernel::run("readingStudent", s2);
  };
  auto alone = [](const Value& r) {
    for (const auto& d : r.at("designs").asArray())
      if (d.at("name").asString() == "its memory alone, no feedback") return d.num("test", 0);
    return 0.0;
  };
  // The golden questions (the library's 60, about its stored capsule): what it remembers must not answer them worse.
  auto golden = [](const std::vector<std::string>& remembered) {
    Value c = Value::object().set("definitionWeight", 1.0);
    if (!remembered.empty()) c.set("memoryRecords", list(remembered));
    Value s2 = Value::object();
    s2.set("capsuleQA", c);
    return kernel::run("capsuleQA", s2).num("f1", 0);
  };
  const Value first = study(records);
  const Value& next = first.at("remember");
  const std::string record = next.at("record").asString();
  std::vector<std::string> with = records;
  with.push_back(record);
  const double before = alone(first), after = alone(study(with));
  const bool heldOut = after >= before - 1e-9;
  const double goldenBefore = heldOut ? golden(records) : 0.0, goldenAfter = heldOut ? golden(with) : 0.0;
  const bool kept = heldOut && goldenAfter >= goldenBefore - kGoldenDrop;
  out.set("train", static_cast<double>(train.size()))
      .set("test", static_cast<double>(test.size()))
      .set("before", before)
      .set("after", after)
      .set("study", next.num("test", 0))
      .set("generation", next.num("generation", 0))
      .set("kept", kept)
      .set("reason", kept      ? "its memory with this study answers the held-out lessons and the golden questions at least as well"
                     : heldOut ? "its memory with this study answers the golden questions worse: not kept"
                               : "its memory with this study answers the held-out lessons worse: not kept");
  if (heldOut) out.set("goldenBefore", goldenBefore).set("goldenAfter", goldenAfter);
  if (kept) out.set("record", record);
  return out.json();
}

std::vector<std::string> blobsOf(const std::string& text) {
  std::vector<std::string> out;
  std::size_t i = text.find_first_not_of(" \t\r\n");
  if (i != std::string::npos && text[i] == '[') {
    for (std::size_t q = text.find('"', i); q != std::string::npos; q = text.find('"', q)) {
      const std::size_t e = text.find('"', q + 1);
      if (e == std::string::npos) break;
      out.push_back(text.substr(q + 1, e - q - 1));
      q = e + 1;
    }
    return out;
  }
  std::istringstream in(text);
  std::string l;
  while (std::getline(in, l)) {
    while (!l.empty() && (l.back() == '\r' || l.back() == ' ')) l.pop_back();
    if (!l.empty()) out.push_back(l);
  }
  return out;
}

int run(const std::vector<std::string>& argsIn, std::vector<std::string> records, std::istream& in, std::ostream& out,
        std::ostream& err) {
  std::vector<std::string> args;
  try {
    for (std::size_t i = 0; i < argsIn.size(); ++i) {
      if (argsIn[i] == "--records" && i + 1 < argsIn.size()) {
        auto blobs = blobsOf(readFile(argsIn[++i]));
        records.insert(records.end(), blobs.begin(), blobs.end());
      } else {
        args.push_back(argsIn[i]);
      }
    }
    const std::string cmd = args.empty() ? "" : args[0];
    if (cmd == "version") {
      std::ostringstream fp;
      fp << std::hex << std::setw(8) << std::setfill('0') << memory::capsuleFingerprint();
      out << "synaptic reader: capsule " << fp.str() << ", " << memory::storedSegments().size() << " stored segment(s)\n";
      return 0;
    }
    if (cmd == "practise") {
      std::vector<Lesson> lessons;
      for (std::string l; std::getline(in, l);) {
        std::vector<std::string> f(1);
        for (char ch : l) {
          if (ch == '\t') f.emplace_back();
          else f.back() += ch;
        }
        if (f.size() >= 3) lessons.push_back({f[0], f[1], std::vector<std::string>(f.begin() + 2, f.end())});
      }
      out << practise(lessons, records, "lessons given to reader practise") << "\n";
      return 0;
    }
    if (cmd == "from" && args.size() >= 2) {
      std::string q;
      for (std::size_t i = 1; i < args.size(); ++i) q += (i > 1 ? " " : "") + args[i];
      std::vector<std::string> passages;
      for (std::string l; std::getline(in, l);) passages.push_back(l.substr(l.rfind('\t') == std::string::npos ? 0 : l.rfind('\t') + 1));
      out << answerFrom(passages, q, records) << "\n";
      return 0;
    }
    const Knowledge k = know(records);
    if (cmd == "info") {
      out << info(k).json() << "\n";
      return 0;
    }
    if (cmd == "ask" && args.size() >= 2) {
      std::string q;
      for (std::size_t i = 1; i < args.size(); ++i) q += (i > 1 ? " " : "") + args[i];
      out << line(answer(k, {q}).at("answers").asArray()[0]) << "\n";
      return 0;
    }
    if (cmd == "session") {
      Value ready = info(k);
      ready.set("ready", true);
      out << ready.json() << std::endl;
      for (std::string l; std::getline(in, l);) {
        while (!l.empty() && (l.back() == '\r' || l.back() == '\n')) l.pop_back();
        try {
          if (l.find_first_not_of(" \t") == std::string::npos) out << Value::object().set("found", false).set("answer", "").json() << std::endl;
          else out << line(answer(k, {l}).at("answers").asArray()[0]) << std::endl;
        } catch (const std::exception& e) {
          out << Value::object().set("found", false).set("answer", "").set("error", e.what()).json() << std::endl;
        }
      }
      return 0;
    }
    err << "usage: reader version | info | ask QUESTION | session | from QUESTION | practise  [--records FILE]\n";
    return 2;
  } catch (const std::exception& e) {
    err << "reader: " << e.what() << "\n";
    return 1;
  }
}

}  // namespace syn::reader
