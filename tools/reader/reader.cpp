// The reader as a native tool (reader.hpp).
#include "reader.hpp"

#include <fstream>
#include <iomanip>
#include <iostream>
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

// What it knows, decoded once.
struct Knowledge {
  Value capsule, report;
  bool stored = false;
  Value records;
};

Knowledge know(const std::vector<std::string>& segments, const std::vector<std::string>& records) {
  Value spec = Value::object();
  if (!segments.empty()) spec.set("segments", list(segments));
  const Value joined = memory::capsuleOf(spec);
  return {joined.at("capsule"), joined.at("segments"), joined.at("stored").asBool(), list(records)};
}

Value info(const Knowledge& k) {
  double facts = 0;
  for (const auto& p : k.capsule.at("passages").asArray()) facts += static_cast<double>(p.at("facts").size());
  const memory::Memory mem = memory::recallFor(Value::object().set("memoryRecords", k.records));
  return Value::object()
      .set("passages", static_cast<double>(k.capsule.at("passages").size()))
      .set("facts", facts)
      .set("words", static_cast<double>(k.capsule.at("dictionary").size()))
      .set("stored", k.stored)
      .set("segments", k.report)
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
// half of what the part asks -- and, where the part asks what something is ("What is X?"), its fact says what
// something is (a form of "be", "means", "is called"), not merely mentions X.
std::string line(const Value& a) {
  Value v = a;
  const bool said = !a.at("answer").asString().empty();
  bool confident = said, found = false;
  for (const auto& p : a.at("parts").asArray()) {
    confident = confident && p.num("covers", 0) >= 0.5;
    if (p.has("definition") && p.at("definition").asBool()) confident = confident && p.has("copula") && p.at("copula").asBool();
    found = found || p.num("covers", 0) > 0;
  }
  v.set("found", said && found).set("confident", said && found && confident);
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
  const Knowledge k{ing.at("capsule"), Value::array(), false, list(records)};
  Value a = Value::object();
  {
    const Value answered = answer(k, {question});
    a = answered.at("answers").asArray()[0];
  }
  // the line, with how many passages it read
  const std::string l = line(a);
  return l.substr(0, l.size() - 1) + ",\"passages\":" + std::to_string(kept.size()) + "}";
}

std::string answerKnown(const std::vector<std::string>& segments, const std::vector<std::string>& records, const std::string& question) {
  const Knowledge k = know(segments, records);
  const Value answered = answer(k, {question});
  return line(answered.at("answers").asArray()[0]);
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

int run(const std::vector<std::string>& argsIn, std::vector<std::string> segments, std::vector<std::string> records, std::istream& in,
        std::ostream& out, std::ostream& err) {
  std::vector<std::string> args;
  bool stored = false;
  try {
    for (std::size_t i = 0; i < argsIn.size(); ++i) {
      if ((argsIn[i] == "--segments" || argsIn[i] == "--records") && i + 1 < argsIn.size()) {
        auto blobs = blobsOf(readFile(argsIn[i + 1]));
        auto& to = argsIn[i] == "--segments" ? segments : records;
        to.insert(to.end(), blobs.begin(), blobs.end());
        ++i;
      } else if (argsIn[i] == "--stored") {
        stored = true;
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
    if (cmd == "ingest" && args.size() >= 3) {
      Value passages = Value::array();
      for (const auto& l : blobsOf(readFile(args[2]))) passages.push(l.substr(l.rfind('\t') == std::string::npos ? 0 : l.rfind('\t') + 1));
      Value c = Value::object().set("passages", passages);
      if (!segments.empty()) c.set("base", list(segments));
      if (stored) c.set("stored", true);
      Value spec = Value::object();
      spec.set("capsuleIngest", c);
      const Value r = kernel::run("capsuleIngest", spec);
      std::ofstream(args[1]) << r.at("segment").asString() << "\n";
      out << Value::object()
                 .set("segment", args[1])
                 .set("passages", r.at("passages"))
                 .set("facts", r.at("facts"))
                 .set("newWords", r.at("newWords"))
                 .set("knownWords", r.at("knownWords"))
                 .set("routed", r.at("routed"))
                 .set("bytes", r.at("bytes"))
                 .set("base", r.at("base"))
                 .json()
          << "\n";
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
    const Knowledge k = know(segments, records);
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
    if (cmd == "batch") {
      std::vector<std::string> qs;
      for (std::string l; std::getline(in, l);)
        if (!l.empty()) qs.push_back(l);
      if (qs.empty()) return 0;
      const Value answered = answer(k, qs);
      for (const auto& a : answered.at("answers").asArray()) out << line(a) << "\n";
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
    err << "usage: reader version | info | ask QUESTION | session | batch | from QUESTION | ingest OUT TSV [--stored]  [--segments FILE] [--records FILE]\n";
    return 2;
  } catch (const std::exception& e) {
    err << "reader: " << e.what() << "\n";
    return 1;
  }
}

}  // namespace syn::reader
