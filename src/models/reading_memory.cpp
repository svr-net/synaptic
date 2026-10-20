// What the reader keeps between runs: its memory of reading, note-taking and constructing habits,
// and its knowledge, a capsule of what it read as points of the space (see reading_memory.hpp).
#include "reading_memory.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <map>
#include <set>

#include "reading_text.hpp"

namespace syn::models::memory {

using namespace text;

const char* const kHabits[kHabitCount] = {"overlap with the question (BM25)", "position in the passage", "first sentence", "length",
                                          "is itself a question", "two sentences", "holds what is asked (a number, a name)",
                                          "covers the question"};
const char* const kNoteHabits[kHabitCount] = {"informative words (idf mass)", "density of informative words", "shares words with the rest",
                                              "position in the passage", "first sentence", "length", "is itself a question",
                                              "holds a number or a name"};
// The constructing habits, the reasoning of an answer: the evidence (how much of the question's
// information the fact holds), then how the answer is said — the whole fact restating its subject,
// the detail next to it, or a part — and whether it holds what is asked and covers the question.
const char* const kConstructHabits[kHabitCount] = {"evidence: the fact holds the question's informative words", "says the detail next to it (two facts)",
                                                   "says the whole fact, restating its subject", "says the predicate (relation, object)", "says the object",
                                                   "says the subject", "holds what is asked (a number, a name)", "covers the question"};
const char* const kRoleNames[kRoles] = {"subject", "relation", "object"};
const double kHabitPrior[kHabitCount] = {1, 0, 0, 0, 0, -1, 0, 0};
const double kNotePrior[kHabitCount] = {1, 0, 0, 0, 0, 0, 0, 0};
// The prior of constructing: the habits Claude learned as a student of the same lessons
// (tools/qa_golden/claude_student) — answer from the evidence, in full sentences restating the
// subject, with the detail next to the fact, holding what is asked and covering the question.
const double kConstructPrior[kHabitCount] = {1, 0.5, 1, 0, 0, 0, 0.5, 1};

namespace {

// The stored memory (written by tools/qa_golden; generation 0 is the prior).
struct Stored {
  int generation;
  double lessons;
  const char* design;
  const char* source;
  double test;
  const char* fingerprint;
  double habits[kHabitCount];
  const char* notesDesign;
  double notesTest;
  double notes[kHabitCount];
  const char* constructDesign;
  double constructTest;
  double construct[kHabitCount];
};
const Stored kStored =
#include "student_memory.inc"
    ;

// The stored capsule: its packed bytes (packCapsule), in base64, line by line.
const char* const kCapsuleBlob[] = {
#include "reader_capsule.inc"
    nullptr};

bool hasDigit(const std::string& s) { return std::any_of(s.begin(), s.end(), [](unsigned char c) { return std::isdigit(c) != 0; }); }
bool hasName(const std::string& s) {
  for (std::size_t i = 1; i + 1 < s.size(); ++i)
    if (s[i - 1] == ' ' && std::isupper(static_cast<unsigned char>(s[i])) && std::islower(static_cast<unsigned char>(s[i + 1]))) return true;
  return false;
}

// How informative each word is, learned from the passages read (BM25's idf), and their sentences.
struct Reading {
  std::vector<std::vector<std::string>> sents;
  std::map<std::string, double> df;
  double n = 0, avg = 1;
  double idf(const std::string& w) const {
    auto it = df.find(w);
    const double d = it == df.end() ? 0.0 : it->second;
    return std::max(0.0, std::log((n - d + 0.5) / (d + 0.5) + 1.0));
  }
};

Reading read(const std::vector<Item>& items) {
  Reading r;
  r.sents.resize(items.size());
  double totalLen = 0, nSent = 0;
  for (std::size_t i = 0; i < items.size(); ++i) {
    r.sents[i] = sentences(items[i].passage);
    std::set<std::string> seen;
    for (const auto& s : r.sents[i]) {
      const auto ws = words(s, true);
      totalLen += static_cast<double>(ws.size());
      nSent += 1;
      seen.insert(ws.begin(), ws.end());
    }
    for (const auto& w : seen) r.df[w] += 1;
  }
  r.n = static_cast<double>(items.size());
  r.avg = totalLen / std::max(1.0, nSent);
  return r;
}

void bestOf(Candidates& p) {
  for (std::size_t c = 1; c < p.reward.size(); ++c)
    if (p.reward[c] > p.reward[p.best]) p.best = c;
}

}  // namespace

std::string fingerprint() {
  std::string f = "habits v1:";
  for (const char* h : kHabits) f += std::string(" ") + h + ";";
  f += " notes v1:";
  for (const char* h : kNoteHabits) f += std::string(" ") + h + ";";
  f += " construct v2:";
  for (const char* h : kConstructHabits) f += std::string(" ") + h + ";";
  return f;
}

std::vector<Candidates> answerCandidates(const std::vector<Item>& items) {
  const Reading r = read(items);
  std::vector<Candidates> out;
  for (std::size_t i = 0; i < items.size(); ++i) {
    const auto& q = items[i].question;
    std::set<std::string> qw;
    for (const auto& w : words(q, true)) qw.insert(w);
    std::string lower = q;
    std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    const bool asksNumber = lower.find("how many") != std::string::npos || lower.find("how much") != std::string::npos ||
                            lower.find("percent") != std::string::npos || lower.find("what year") != std::string::npos ||
                            lower.rfind("when", 0) == 0 || lower.find("how long") != std::string::npos || lower.find("how old") != std::string::npos;
    const bool asksName = lower.rfind("who", 0) == 0 || lower.find(" who ") != std::string::npos || lower.find("which person") != std::string::npos;
    const auto& ss = r.sents[i];
    Candidates p;
    std::vector<int> pos;
    for (std::size_t k = 0; k < ss.size() && p.texts.size() < 40; ++k) {
      p.texts.push_back(ss[k]);
      pos.push_back(static_cast<int>(k));
      p.pair.push_back(false);
    }
    for (std::size_t k = 0; k + 1 < ss.size() && p.texts.size() < 64; ++k) {
      p.texts.push_back(ss[k] + " " + ss[k + 1]);
      pos.push_back(static_cast<int>(k));
      p.pair.push_back(true);
    }
    double maxBm = 1e-9;
    std::vector<double> bm(p.texts.size());
    for (std::size_t c = 0; c < p.texts.size(); ++c) {
      const auto ws = words(p.texts[c], true);
      std::map<std::string, double> tf;
      for (const auto& w : ws) tf[w] += 1;
      const double norm = 1.2 * (1 - 0.75 + 0.75 * static_cast<double>(ws.size()) / r.avg);
      double e = 0;
      for (const auto& w : qw)
        if (tf.count(w)) e += r.idf(w) * tf[w] * 2.2 / (tf[w] + norm);
      bm[c] = e;
      if (!p.pair[c]) maxBm = std::max(maxBm, e);
    }
    for (std::size_t c = 0; c < p.texts.size(); ++c) {
      const auto ws = words(p.texts[c], true);
      std::set<std::string> have(ws.begin(), ws.end());
      double covered = 0;
      for (const auto& w : qw) covered += have.count(w) ? 1.0 : 0.0;
      const double fit = (asksNumber && hasDigit(p.texts[c])) || (asksName && hasName(p.texts[c])) ? 1.0 : 0.0;
      p.habits.push_back({bm[c] / maxBm, ss.size() > 1 ? double(pos[c]) / double(ss.size() - 1) : 0.0, pos[c] == 0 && !p.pair[c] ? 1.0 : 0.0,
                          static_cast<double>(words(p.texts[c], false).size()) / 40.0, p.texts[c].find('?') != std::string::npos ? 1.0 : 0.0,
                          p.pair[c] ? 1.0 : 0.0, fit, qw.empty() ? 0.0 : covered / static_cast<double>(qw.size())});
      p.reward.push_back(items[i].golden.empty() ? 0.0 : tokenF1(p.texts[c], items[i].golden));
    }
    bestOf(p);
    out.push_back(std::move(p));
  }
  return out;
}

std::vector<Candidates> noteCandidates(const std::vector<Item>& items) {
  const Reading r = read(items);
  std::vector<Candidates> out;
  for (std::size_t i = 0; i < items.size(); ++i) {
    const auto& ss = r.sents[i];
    const std::size_t n = std::min<std::size_t>(ss.size(), 64);
    std::vector<std::set<std::string>> bags(n);
    for (std::size_t k = 0; k < n; ++k)
      for (const auto& w : words(ss[k], true)) bags[k].insert(w);
    Candidates p;
    for (std::size_t k = 0; k < n; ++k) {
      double mass = 0;
      for (const auto& w : bags[k]) mass += r.idf(w);
      const double len = static_cast<double>(words(ss[k], true).size());
      std::set<std::string> others;
      for (std::size_t j = 0; j < n; ++j)
        if (j != k) others.insert(bags[j].begin(), bags[j].end());
      double shared = 0;
      for (const auto& w : bags[k]) shared += others.count(w) ? 1.0 : 0.0;
      p.texts.push_back(ss[k]);
      p.pair.push_back(false);
      p.habits.push_back({mass / 20.0, mass / std::max(1.0, len) / 5.0, bags[k].empty() ? 0.0 : shared / static_cast<double>(bags[k].size()),
                          n > 1 ? double(k) / double(n - 1) : 0.0, k == 0 ? 1.0 : 0.0, static_cast<double>(words(ss[k], false).size()) / 40.0,
                          ss[k].find('?') != std::string::npos ? 1.0 : 0.0, hasDigit(ss[k]) || hasName(ss[k]) ? 1.0 : 0.0});
      p.reward.push_back(items[i].golden.empty() ? 0.0 : tokenF1(ss[k], items[i].golden));
    }
    bestOf(p);
    out.push_back(std::move(p));
  }
  return out;
}

const char* const kSectionNames[kSections] = {"answer", "notes", "construct"};

namespace {

// The fingerprint of one section of the habits.
std::string sectionOf(const std::string& f, int section) {
  const char* from[] = {"habits v", " notes v", " construct v"};
  const char* to[] = {" notes v", " construct v", nullptr};
  const auto a = f.find(from[section]);
  if (a == std::string::npos) return std::string();
  const auto b = to[section] ? f.find(to[section], a) : std::string::npos;
  return f.substr(a, b == std::string::npos ? std::string::npos : b - a);
}
std::uint32_t hash32(const std::string& s) {
  std::uint32_t h = 2166136261u;
  for (unsigned char ch : s) h = (h ^ ch) * 16777619u;
  return h;
}

}  // namespace

namespace {

constexpr std::uint8_t kRecordMagic[4] = {'S', 'Y', 'N', 'M'};
constexpr std::uint32_t kRecordVersion = 1;

void putVar(std::string& out, std::uint32_t v) {
  while (v >= 0x80) {
    out.push_back(static_cast<char>(v | 0x80));
    v >>= 7;
  }
  out.push_back(static_cast<char>(v));
}
std::uint32_t getVar(const std::string& in, std::size_t& at) {
  std::uint32_t v = 0;
  for (int shift = 0; at < in.size() && shift < 35; shift += 7) {
    const auto b = static_cast<std::uint8_t>(in[at++]);
    v |= static_cast<std::uint32_t>(b & 0x7f) << shift;
    if (!(b & 0x80)) break;
  }
  return v;
}
void put32(std::string& out, std::uint32_t v) {
  for (int k = 0; k < 4; ++k) out.push_back(static_cast<char>(v >> (8 * k)));
}
std::uint32_t get32(const std::string& in, std::size_t& at) {
  std::uint32_t v = 0;
  for (int k = 0; k < 4 && at < in.size(); ++k) v |= std::uint32_t{static_cast<std::uint8_t>(in[at++])} << (8 * k);
  return v;
}
void putF(std::string& out, double d) {
  const auto f = static_cast<float>(d);
  std::uint32_t v;
  std::memcpy(&v, &f, 4);
  put32(out, v);
}
double getF(const std::string& in, std::size_t& at) {
  const std::uint32_t v = get32(in, at);
  float f;
  std::memcpy(&f, &v, 4);
  return f;
}
void putStr(std::string& out, const std::string& s) {
  putVar(out, static_cast<std::uint32_t>(s.size()));
  out += s;
}
std::string getStr(const std::string& in, std::size_t& at) {
  const std::uint32_t n = getVar(in, at);
  std::string s = in.substr(std::min(at, in.size()), n);
  at += n;
  return s;
}
const char* const kB64 = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
std::string b64(const std::string& b) {
  std::string out;
  for (std::size_t i = 0; i < b.size(); i += 3) {
    const auto at = [&](std::size_t k) { return k < b.size() ? std::uint32_t{static_cast<std::uint8_t>(b[k])} : 0u; };
    const std::uint32_t n = (at(i) << 16) | (at(i + 1) << 8) | at(i + 2);
    out += kB64[(n >> 18) & 63];
    out += kB64[(n >> 12) & 63];
    out += i + 1 < b.size() ? kB64[(n >> 6) & 63] : '=';
    out += i + 2 < b.size() ? kB64[n & 63] : '=';
  }
  return out;
}
std::string unb64(const std::string& s) {
  std::string out;
  std::uint32_t acc = 0;
  int have = 0;
  for (char c : s) {
    const char* at = std::strchr(kB64, c);
    if (!at || c == '\0') continue;
    acc = (acc << 6) | static_cast<std::uint32_t>(at - kB64);
    have += 6;
    if (have >= 8) {
      have -= 8;
      out.push_back(static_cast<char>((acc >> have) & 0xff));
    }
  }
  return out;
}

// The stored memory with its records folded in, in order.
Memory withRecords(Memory m, const Value& records) {
  if (!records.isArray() && !(records.isNumbers() && records.size() == 0)) return m;
  int last[kSections] = {m.used ? m.generation : 0, m.used ? m.generation : 0, m.used ? m.generation : 0};
  for (std::size_t i = 0; i < records.size(); ++i) {
    std::string error;
    const Record r = unpackRecord(records[i].asString(), &error);
    if (error.empty() && r.generation <= last[r.section])
      error = "generation " + std::to_string(r.generation) + " does not follow generation " + std::to_string(last[r.section]) + " of its section";
    Value row = Value::object().set("used", error.empty());
    if (error.empty()) {
      double* w = r.section == 0 ? m.habits : r.section == 1 ? m.notes : m.construct;
      double next[kHabitCount];
      fold(w, m.evidence[r.section], r, next);
      std::copy(next, next + kHabitCount, w);
      m.evidence[r.section] += r.lessons;
      last[r.section] = r.generation;
      m.generation = std::max(m.generation, r.generation);
      m.lessons += r.lessons;
      m.used = true;
      (r.section == 0 ? m.test : r.section == 1 ? m.notesTest : m.constructTest) = r.test;
      (r.section == 0 ? m.design : r.section == 1 ? m.notesDesign : m.constructDesign) = r.design;
      row.set("section", kSectionNames[r.section]).set("generation", r.generation).set("lessons", r.lessons).set("test", r.test).set("source", r.source);
    } else {
      row.set("reason", error);
    }
    m.records.push(row);
  }
  return m;
}

}  // namespace

std::string packRecord(const Record& r) {
  std::string b(reinterpret_cast<const char*>(kRecordMagic), 4);
  putVar(b, kRecordVersion);
  b.push_back(static_cast<char>(r.section));
  put32(b, hash32(sectionOf(fingerprint(), r.section)));
  putVar(b, static_cast<std::uint32_t>(std::max(0, r.generation)));
  putVar(b, static_cast<std::uint32_t>(std::max(0.0, std::round(r.lessons))));
  putF(b, r.test);
  for (int k = 0; k < kHabitCount; ++k) putF(b, r.weights[k]);
  putStr(b, r.design);
  putStr(b, r.source);
  put32(b, hash32(b));
  return b64(b);
}

Record unpackRecord(const std::string& base64, std::string* error) {
  Record r;
  const std::string b = unb64(base64);
  auto fail = [&](const char* why) {
    if (error) *error = why;
    return r;
  };
  if (b.size() < 12 || b.compare(0, 4, reinterpret_cast<const char*>(kRecordMagic), 4) != 0) return fail("not a memory record");
  std::size_t sumAt = b.size() - 4;
  if (get32(b, sumAt) != hash32(b.substr(0, b.size() - 4))) return fail("its checksum does not match (damaged)");
  std::size_t at = 4;
  if (getVar(b, at) != kRecordVersion) return fail("written by another version of the record format");
  r.section = static_cast<std::uint8_t>(b[at++]);
  if (r.section < 0 || r.section >= kSections) return fail("an unknown section");
  if (get32(b, at) != hash32(sectionOf(fingerprint(), r.section))) return fail("learned for other habits (its section's fingerprint differs)");
  r.generation = static_cast<int>(getVar(b, at));
  r.lessons = getVar(b, at);
  r.test = getF(b, at);
  for (int k = 0; k < kHabitCount; ++k) r.weights[k] = getF(b, at);
  r.design = getStr(b, at);
  r.source = getStr(b, at);
  if (at != b.size() - 4) return fail("its parts do not add up (damaged)");
  if (error) error->clear();
  return r;
}

Memory recallFor(const Value& spec) {
  return recall(!spec.flag("memory", true), spec.str("memoryFingerprint", ""), spec.has("memoryRecords") ? spec.at("memoryRecords") : Value());
}

void fold(const double* weights, double evidence, const Record& r, double* out) {
  const double n = std::max(0.0, r.lessons), total = evidence + n;
  for (int k = 0; k < kHabitCount; ++k) out[k] = total > 0 ? (evidence * weights[k] + n * r.weights[k]) / total : weights[k];
}

Memory recall(bool fresh, const std::string& fingerprintOf, const Value& records) {
  Memory m;
  std::copy(kHabitPrior, kHabitPrior + kHabitCount, m.habits);
  std::copy(kNotePrior, kNotePrior + kHabitCount, m.notes);
  std::copy(kConstructPrior, kConstructPrior + kHabitCount, m.construct);
  if (fresh) {
    m.reason = "started fresh";
    return m;
  }
  // Each set of habits is used only if it was learned for these habits: the fingerprint, section by
  // section (reading, notes, constructing).
  const std::string stored = fingerprintOf.empty() ? kStored.fingerprint : fingerprintOf;
  const std::string now = fingerprint();
  auto section = [](const std::string& f, const char* from, const char* to) {
    const auto a = f.find(from);
    if (a == std::string::npos) return std::string();
    const auto b = to ? f.find(to, a) : std::string::npos;
    return f.substr(a, b == std::string::npos ? std::string::npos : b - a);
  };
  const bool habits = section(stored, "habits v", " notes v") == section(now, "habits v", " notes v");
  const bool notes = section(stored, " notes v", " construct v") == section(now, " notes v", " construct v");
  const bool construct = section(stored, " construct v", nullptr) == section(now, " construct v", nullptr);
  if (!habits && !notes && !construct) {
    m.reason = "the memory was learned for other habits (fingerprint differs); started from the prior";
    return withRecords(m, records);
  }
  m.used = true;
  if (!habits || !notes || !construct)
    m.reason = std::string("partly: ") + (habits ? "" : "reading habits ") + (notes ? "" : "note-taking habits ") +
               (construct ? "" : "constructing habits ") + "were learned for other habits; those start from the prior";
  m.generation = kStored.generation;
  m.lessons = kStored.lessons;
  m.source = kStored.source;
  if (habits) {
    m.design = kStored.design;
    m.test = kStored.test;
    std::copy(kStored.habits, kStored.habits + kHabitCount, m.habits);
  }
  if (notes) {
    m.notesDesign = kStored.notesDesign;
    m.notesTest = kStored.notesTest;
    std::copy(kStored.notes, kStored.notes + kHabitCount, m.notes);
  }
  if (construct) {
    m.constructDesign = kStored.constructDesign;
    m.constructTest = kStored.constructTest;
    std::copy(kStored.construct, kStored.construct + kHabitCount, m.construct);
  }
  for (int sct = 0; sct < kSections; ++sct) m.evidence[sct] = (sct == 0 ? habits : sct == 1 ? notes : construct) ? kStored.lessons : 0.0;
  return withRecords(m, records);
}

// ---- The grammar.

namespace {

bool isArticle(const std::string& w) {
  std::string l = w;
  std::transform(l.begin(), l.end(), l.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return l == "a" || l == "an" || l == "the";
}
std::string lower(std::string w) {
  std::transform(w.begin(), w.end(), w.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return w;
}
const std::set<std::string>& auxiliaries() {
  static const std::set<std::string> s = {"is", "are", "was", "were", "be", "been", "being", "has", "have", "had", "can", "could", "will",
                                          "would", "may", "might", "must", "should", "shall", "do", "does", "did"};
  return s;
}
// Words that go with a verb in its relation ("is used to", "can also help").
bool followsVerb(const std::string& w) {
  static const std::set<std::string> s = {"not", "also", "often", "usually", "still", "only", "then", "used", "able", "to", "by", "as",
                                          "of", "in", "on", "for", "with", "from", "into", "than"};
  const std::string l = lower(w);
  return auxiliaries().count(l) || s.count(l);
}
bool startsClause(const std::string& w) {
  static const std::set<std::string> s = {"and", "but", "which", "while", "so", "because", "whereas"};
  return s.count(lower(w)) > 0;
}

// The clauses of a sentence, as words (articles dropped): split at ; and :, and at a comma before
// a conjunction or "which".
std::vector<std::vector<std::string>> clausesOf(const std::string& sentence) {
  std::vector<std::vector<std::string>> out(1);
  std::string w;
  bool comma = false;
  auto flush = [&]() {
    if (w.empty()) return;
    while (!w.empty() && !std::isalnum(static_cast<unsigned char>(w.back())) && w.back() != '%') w.pop_back();
    if (!w.empty() && !isArticle(w)) {
      if (comma && startsClause(w) && !out.back().empty()) out.emplace_back();
      out.back().push_back(w);
    }
    comma = false;
    w.clear();
  };
  for (std::size_t i = 0; i < sentence.size(); ++i) {
    const char ch = sentence[i];
    const bool inner = (ch == '\'' || ch == '-' || ch == '.' || ch == '%') && !w.empty() && i + 1 < sentence.size() &&
                       (std::isalnum(static_cast<unsigned char>(sentence[i + 1])) || ch == '%');
    if (std::isalnum(static_cast<unsigned char>(ch)) || inner) {
      w += ch;
      continue;
    }
    flush();
    if (ch == ';' || ch == ':') {
      if (!out.back().empty()) out.emplace_back();
    } else if (ch == ',') {
      comma = true;
    }
  }
  flush();
  return out;
}

std::vector<std::string> wordsOf(const Fact& f) {
  std::vector<std::string> w = f.subject;
  w.insert(w.end(), f.relation.begin(), f.relation.end());
  w.insert(w.end(), f.object.begin(), f.object.end());
  return w;
}
std::string joined(const std::vector<std::string>& ws) {
  std::string s;
  for (const auto& w : ws) s += (s.empty() ? "" : " ") + w;
  return s;
}

}  // namespace

std::vector<Clause> clauses(const std::string& passage) {
  // What a reader knows of each word on its own — its word class's closed lists (auxiliaries,
  // prepositions and determiners, the words that go with a verb), its ending, its capital — as
  // cues; which word is the verb is the geometry's to decide (capsuleIngest).
  static const std::set<std::string> before = {"of", "with", "in", "on", "for", "by", "to", "from", "other", "many", "some", "all",
                                               "these", "those", "their", "its", "his", "her", "our", "your", "and", "or"};
  std::vector<Clause> out;
  for (const auto& sentence : sentences(passage))
    for (const auto& t : clausesOf(sentence)) {
      if (t.size() < 3) continue;
      Clause c;
      c.words = t;
      const std::string first = lower(t[0]);
      for (std::size_t i = 0; i < t.size(); ++i) {
        const std::string l = lower(t[i]);
        std::uint32_t cue = 0;
        if (auxiliaries().count(l)) cue |= kCueAuxiliary;
        if ((l.size() > 3 && l.compare(l.size() - 2, 2, "ed") == 0) || (l.size() > 3 && l.back() == 's' && l[l.size() - 2] != 's')) cue |= kCueVerbEnding;
        if (text::stopwords().count(l)) cue |= kCueFunctionWord;
        if (std::isupper(static_cast<unsigned char>(t[i][0]))) cue |= kCueCapital;
        if (i > 0 && before.count(lower(t[i - 1]))) cue |= kCueAfterPreposition;
        if (i == 0) cue |= kCueFirst;
        if (followsVerb(t[i])) cue |= kCueGoesWithVerb;
        if (i == 1 && first.size() > 3 && first.back() == 's') cue |= kCueAfterPlural;
        c.cues.push_back(cue);
      }
      out.push_back(std::move(c));
    }
  return out;
}

void wordCoordinates(const std::string& word, float u[3]) {
  // FNV-1a 64, finished by the 64-bit mixer, cut into three 16-bit coordinates: 48 bits of the word.
  std::uint64_t h = 14695981039346656037ull;
  for (unsigned char c : word) h = (h ^ c) * 1099511628211ull;
  h ^= h >> 33;
  h *= 0xff51afd7ed558ccdull;
  h ^= h >> 33;
  h *= 0xc4ceb9fe1a85ec53ull;
  h ^= h >> 33;
  for (int k = 0; k < 3; ++k) u[k] = static_cast<float>((h >> (16 * k)) & 0xffffu) / 65535.0f * 4.0f - 2.0f;
}

const char* grammarWgsl() {
  return R"wgsl(
// The grammar as operators of the space. A word at position k of a fact, in grammatical role r, is
// its point turned by the rotor R_r · S^k: R_r turns the (N, A) plane by the role's angle, S^k the
// (B, C) plane by k steps. Word order and role live in the geometry, not in how points are stored.
const kGrammarRole = 2.0943951;
const kGrammarStep = 0.19634954;
// A rotor of the grammar in plane (i, i + 1), its cosine and sine normalised so that it keeps the
// point on S³ exactly (GPU sin and cos are approximate).
fn grammarTurn(p : V5, i : u32, alpha : f32) -> V5 {
  let cs = normalize(vec2<f32>(cos(alpha), sin(alpha)));
  var r = p;
  r.x[i] = cs.x * p.x[i] - cs.y * p.x[i + 1u];
  r.x[i + 1u] = cs.y * p.x[i] + cs.x * p.x[i + 1u];
  return r;
}
fn bindWord(p : V5, role : u32, position : u32) -> V5 {
  return grammarTurn(grammarTurn(p, 0u, 0.5 + kGrammarRole * f32(role)), 2u, 0.3 + kGrammarStep * f32(position));
}
// The grammar's rotor that turns the point w to x, read off the two rotation angles:
// role · 32 + position, or −1 if no rotor of the grammar does (the norms in each plane must agree).
fn grammarOf(x : V5, w : V5) -> i32 {
  let a = vec2<f32>(x.x[0], x.x[1]);
  let b = vec2<f32>(w.x[0], w.x[1]);
  let c = vec2<f32>(x.x[2], x.x[3]);
  let d = vec2<f32>(w.x[2], w.x[3]);
  if (abs(length(a) - length(b)) > 1e-3 || abs(length(c) - length(d)) > 1e-3 || abs(x.e - w.e) > 1e-3) { return -1; }
  if (length(b) < 1e-3 || length(d) < 1e-3) { return -1; }
  let u = (atan2(a.y, a.x) - atan2(b.y, b.x) - 0.5) / kGrammarRole;
  let v = (atan2(c.y, c.x) - atan2(d.y, d.x) - 0.3) / kGrammarStep;
  if (abs(u - round(u)) > 0.01 || abs(v - round(v)) > 0.05) { return -1; }
  let r = ((i32(round(u)) % 3) + 3) % 3;
  let k = ((i32(round(v)) % 32) + 32) % 32;
  return r * 32 + k;
}
)wgsl";
}

std::string realize(const std::vector<std::string>& words) {
  // "." ends a sentence of the answer; each sentence starts with a capital.
  std::string s;
  bool start = true;
  for (const auto& w : words) {
    if (w == ".") {
      if (!s.empty()) s += ".";
      start = true;
      continue;
    }
    std::string x = w;
    if (start) x[0] = static_cast<char>(std::toupper(static_cast<unsigned char>(x[0])));
    s += (s.empty() ? "" : " ") + x;
    start = false;
  }
  return s.empty() ? s : s + ".";
}

std::vector<std::vector<Fact>> factsOf(const Value& recalled) {
  std::vector<std::vector<Fact>> out;
  const Value& dict = recalled.at("dictionary");
  const Value& passages = recalled.at("passages");
  for (std::size_t p = 0; p < passages.size(); ++p) {
    std::vector<Fact> facts;
    const Value& fs = passages.asArray()[p].at("facts");
    for (std::size_t i = 0; i < fs.size(); ++i) {
      Fact f;
      std::vector<std::string>* parts[3] = {&f.subject, &f.relation, &f.object};
      for (int r = 0; r < kRoles; ++r) {
        const Value& ids = fs.asArray()[i].at(kRoleNames[r]);
        for (std::size_t k = 0; k < ids.size(); ++k) parts[r]->push_back(dict[static_cast<std::size_t>(ids[k].asNumber())].asString());
      }
      facts.push_back(std::move(f));
    }
    out.push_back(std::move(facts));
  }
  return out;
}

std::vector<std::string> stemsOf(const std::string& word) { return words(word, true); }

std::vector<QuestionPart> questionParts(const std::string& question) {
  // A second question word after "and" starts a new part; its pronouns refer to the first part.
  static const std::set<std::string> asking = {"what", "how", "why", "who", "when", "where", "which", "whose", "do", "does", "did",
                                               "is", "are", "can", "could"};
  static const std::set<std::string> pronouns = {"they", "their", "them", "it", "its", "this", "these", "those", "he", "she", "his", "her"};
  std::vector<std::string> all;
  std::string w;
  auto flush = [&]() {
    if (!w.empty()) all.push_back(w);
    w.clear();
  };
  for (char ch : question) {
    if (ch == ' ') flush();
    else w += ch;
  }
  flush();
  std::vector<std::string> texts(1);
  for (std::size_t i = 0; i < all.size(); ++i) {
    const std::string l = lower(all[i]);
    if (i > 0 && lower(all[i - 1]) == "and" && asking.count(l) && !texts.back().empty()) {
      texts.back() = texts.back().substr(0, texts.back().size() - 4);  // drop " and"
      texts.emplace_back();
    }
    texts.back() += (texts.back().empty() ? "" : " ") + all[i];
  }
  std::vector<QuestionPart> out;
  for (const auto& t : texts) {
    QuestionPart p;
    p.text = t;
    p.stems = words(t, true);
    const std::string lq = lower(t);
    p.asksNumber = lq.find("how many") != std::string::npos || lq.find("how much") != std::string::npos || lq.find("percent") != std::string::npos ||
                   lq.find("what year") != std::string::npos || lq.rfind("when", 0) == 0 || lq.find("how long") != std::string::npos ||
                   lq.find("how old") != std::string::npos;
    p.asksName = lq.rfind("who", 0) == 0 || lq.find(" who ") != std::string::npos || lq.find("which person") != std::string::npos;
    for (const char* d : {"what is ", "what are ", "what was ", "what were ", "who is ", "who was ", "who were ", "who are ", "what does ",
                          "define ", "what's "})
      p.asksDefinition = p.asksDefinition || lq.rfind(d, 0) == 0;
    // "what does X do / mean": only "mean" asks what it is
    if (lq.rfind("what does ", 0) == 0) p.asksDefinition = lq.find(" mean") != std::string::npos;
    if (!out.empty()) {
      bool refers = false;
      for (const auto& x : words(t, false)) refers = refers || pronouns.count(x);
      if (refers)
        for (const auto& st : out.front().stems)
          if (std::find(p.stems.begin(), p.stems.end(), st) == p.stems.end()) p.referred.push_back(st);
    }
    out.push_back(std::move(p));
  }
  return out;
}

namespace {

// The constructing features of saying `said` (words) from fact k: the shared definition of the
// student's lessons and of capsuleQA's kernel (which computes the same in the space).
std::vector<double> constructFeatures(const QuestionPart& q, const std::vector<std::string>& fact, const std::vector<std::string>& said, int pattern,
                                      const std::function<double(const std::string&)>& idf) {
  std::set<std::string> asked(q.stems.begin(), q.stems.end());
  asked.insert(q.referred.begin(), q.referred.end());
  std::set<std::string> inFact, inSaid;
  for (const auto& w : fact)
    for (const auto& st : stemsOf(w)) inFact.insert(st);
  bool fit = false;
  for (std::size_t k = 0; k < said.size(); ++k) {
    for (const auto& st : stemsOf(said[k])) inSaid.insert(st);
    const bool digit = hasDigit(said[k]);
    const bool name = k > 0 && std::isupper(static_cast<unsigned char>(said[k][0]));
    fit = fit || (q.asksNumber && digit) || (q.asksName && name);
  }
  double mass = 0, held = 0, covered = 0;
  for (const auto& st : asked) {
    mass += idf(st);
    held += inFact.count(st) ? idf(st) : 0.0;
    covered += inSaid.count(st) ? 1.0 : 0.0;
  }
  return {mass > 0 ? held / mass : 0.0, pattern == 4 ? 1.0 : 0.0, pattern == 0 || pattern == 4 ? 1.0 : 0.0, pattern == 1 ? 1.0 : 0.0,
          pattern == 2 ? 1.0 : 0.0, pattern == 3 ? 1.0 : 0.0, fit ? 1.0 : 0.0, asked.empty() ? 0.0 : covered / static_cast<double>(asked.size())};
}

}  // namespace

std::vector<Candidates> constructCandidates(const std::vector<Item>& items, const std::vector<std::vector<Fact>>& facts) {
  // How informative each word is, learned from the facts it recalls (each fact a document).
  std::map<std::string, double> df;
  double n = 0;
  for (const auto& fs : facts)
    for (const auto& f : fs) {
      std::set<std::string> seen;
      for (const auto& w : wordsOf(f))
        for (const auto& st : stemsOf(w)) seen.insert(st);
      for (const auto& st : seen) df[st] += 1;
      n += 1;
    }
  auto idf = [&](const std::string& w) {
    auto it = df.find(w);
    const double d = it == df.end() ? 0.0 : it->second;
    return std::max(0.0, std::log((n - d + 0.5) / (d + 0.5) + 1.0));
  };
  std::vector<Candidates> out;
  for (std::size_t i = 0; i < items.size(); ++i) {
    // A lesson is the whole question as one part (the student learns how to say one answer).
    QuestionPart q;
    q.text = items[i].question;
    q.stems = words(items[i].question, true);
    const auto parts = questionParts(items[i].question);
    for (const auto& pt : parts) {
      q.asksNumber = q.asksNumber || pt.asksNumber;
      q.asksName = q.asksName || pt.asksName;
    }
    const auto& fs = facts[i];
    const std::size_t nf = std::min<std::size_t>(fs.size(), 24);
    Candidates p;
    for (std::size_t k = 0; k < nf; ++k) {
      const Fact& f = fs[k];
      const std::vector<std::string> whole = wordsOf(f);
      std::vector<std::string> predicate = f.relation;
      predicate.insert(predicate.end(), f.object.begin(), f.object.end());
      std::vector<std::string> two = whole;
      if (k + 1 < nf) {
        const auto after = wordsOf(fs[k + 1]);
        two.push_back(".");
        two.insert(two.end(), after.begin(), after.end());
      }
      // The grammar's ways of saying it: the whole fact, its predicate, its object, its subject, and
      // the whole fact with the next one.
      const std::vector<std::string>* said[5] = {&whole, &predicate, &f.object, &f.subject, &two};
      for (int pattern = 0; pattern < 5; ++pattern) {
        if (pattern == 4 && k + 1 >= nf) continue;
        if (said[pattern]->empty() || (pattern == 1 && f.relation.empty())) continue;
        std::vector<std::string> words_;
        for (const auto& w : *said[pattern])
          if (w != ".") words_.push_back(w);
        const std::string t = realize(*said[pattern]);
        p.texts.push_back(t);
        p.pair.push_back(false);
        p.habits.push_back(constructFeatures(q, whole, words_, pattern, idf));
        p.reward.push_back(items[i].golden.empty() ? 0.0 : tokenF1(t, items[i].golden));
      }
    }
    bestOf(p);
    out.push_back(std::move(p));
  }
  return out;
}

std::vector<Fact> factsFor(const std::string& question, const std::vector<std::vector<Fact>>& all, std::size_t keep) {
  std::set<std::string> qw;
  for (const auto& w : words(question, true)) qw.insert(w);
  std::vector<std::pair<double, const Fact*>> ranked;
  for (const auto& fs : all)
    for (const auto& f : fs) {
      const auto ws = words(joined(wordsOf(f)), true);
      double shared = 0;
      for (const auto& w : std::set<std::string>(ws.begin(), ws.end())) shared += qw.count(w) ? 1.0 : 0.0;
      ranked.push_back({shared, &f});
    }
  std::stable_sort(ranked.begin(), ranked.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
  std::vector<Fact> out;
  for (std::size_t k = 0; k < ranked.size() && k < keep; ++k) out.push_back(*ranked[k].second);
  return out;
}

// ---- Packing the capsule: the codes of the index router, bit by bit.

namespace {

void putVarint(std::vector<std::uint8_t>& out, std::uint32_t v) {
  while (v >= 0x80) {
    out.push_back(static_cast<std::uint8_t>(v | 0x80));
    v >>= 7;
  }
  out.push_back(static_cast<std::uint8_t>(v));
}
std::uint32_t getVarint(const std::vector<std::uint8_t>& in, std::size_t& at) {
  std::uint32_t v = 0;
  for (int shift = 0; at < in.size(); shift += 7) {
    const std::uint8_t b = in[at++];
    v |= static_cast<std::uint32_t>(b & 0x7f) << shift;
    if (!(b & 0x80)) break;
  }
  return v;
}
// Bits per code of a passage: the codebooks of its levels — its vocabulary × the grammar's
// 3 roles × 32 places.
unsigned codeBits(std::size_t vocabulary) {
  unsigned b = 1;
  while ((std::size_t{1} << b) < std::max<std::size_t>(2, vocabulary * 96)) ++b;
  return b;
}
const char* const kBase64 = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

}  // namespace

namespace {

constexpr std::uint8_t kMagic[4] = {'S', 'Y', 'N', 'K'};
constexpr std::uint32_t kSegmentVersion = 1;
constexpr unsigned kCellBits = 12;  // the index's 16³ cells

std::uint32_t fnv(const std::uint8_t* b, std::size_t n) {
  std::uint32_t h = 2166136261u;
  for (std::size_t i = 0; i < n; ++i) h = (h ^ b[i]) * 16777619u;
  return h;
}
void putU32(std::vector<std::uint8_t>& out, std::uint32_t v) {
  for (int k = 0; k < 4; ++k) out.push_back(static_cast<std::uint8_t>(v >> (8 * k)));
}
std::uint32_t getU32(const std::vector<std::uint8_t>& in, std::size_t& at) {
  std::uint32_t v = 0;
  for (int k = 0; k < 4 && at < in.size(); ++k) v |= std::uint32_t{in[at++]} << (8 * k);
  return v;
}
// Bit-packing: values of `bits` bits, least significant first.
struct BitWriter {
  std::vector<std::uint8_t>& b;
  std::uint64_t acc = 0;
  unsigned have = 0;
  void put(std::uint64_t v, unsigned bits) {
    acc |= v << have;
    have += bits;
    while (have >= 8) {
      b.push_back(static_cast<std::uint8_t>(acc & 0xff));
      acc >>= 8;
      have -= 8;
    }
  }
  void flush() {
    if (have > 0) b.push_back(static_cast<std::uint8_t>(acc & 0xff));
    acc = 0;
    have = 0;
  }
};
struct BitReader {
  const std::vector<std::uint8_t>& b;
  std::size_t& at;
  std::uint64_t acc = 0;
  unsigned have = 0;
  std::uint64_t get(unsigned bits) {
    while (have < bits && at < b.size()) {
      acc |= static_cast<std::uint64_t>(b[at++]) << have;
      have += 8;
    }
    const std::uint64_t v = acc & ((std::uint64_t{1} << bits) - 1);
    acc >>= bits;
    have -= std::min(have, bits);
    return v;
  }
};
std::string toBase64(const std::vector<std::uint8_t>& b) {
  std::string out;
  for (std::size_t i = 0; i < b.size(); i += 3) {
    const std::uint32_t n = (std::uint32_t{b[i]} << 16) | (i + 1 < b.size() ? std::uint32_t{b[i + 1]} << 8 : 0u) | (i + 2 < b.size() ? b[i + 2] : 0u);
    out += kBase64[(n >> 18) & 63];
    out += kBase64[(n >> 12) & 63];
    out += i + 1 < b.size() ? kBase64[(n >> 6) & 63] : '=';
    out += i + 2 < b.size() ? kBase64[n & 63] : '=';
  }
  return out;
}
std::vector<std::uint8_t> fromBase64(const std::string& base64) {
  std::vector<std::uint8_t> b;
  std::uint32_t acc = 0;
  int have = 0;
  for (char c : base64) {
    const char* at = std::strchr(kBase64, c);
    if (!at || c == '\0') continue;  // '=' and anything else end or pad
    acc = (acc << 6) | static_cast<std::uint32_t>(at - kBase64);
    have += 6;
    if (have >= 8) {
      have -= 8;
      b.push_back(static_cast<std::uint8_t>((acc >> have) & 0xff));
    }
  }
  return b;
}

// One segment read: what it adds, or why it cannot be read.
struct Read {
  std::string error;
  std::uint32_t baseWords = 0, basePassages = 0;
  std::vector<std::string> words;
  std::vector<double> cells;
  Value passages = Value::array();
  std::size_t bytes = 0;
};

Read readSegment(const std::string& base64) {
  Read r;
  const std::vector<std::uint8_t> b = fromBase64(base64);
  r.bytes = b.size();
  if (b.size() < 12 || !std::equal(kMagic, kMagic + 4, b.begin())) {
    r.error = "not a capsule segment";
    return r;
  }
  std::size_t end = b.size() - 4, sumAt = end;
  if (getU32(b, sumAt) != fnv(b.data(), end)) {
    r.error = "its checksum does not match (damaged)";
    return r;
  }
  std::size_t at = 4;
  if (getVarint(b, at) != kSegmentVersion) {
    r.error = "written by another version of the segment format";
    return r;
  }
  if (getU32(b, at) != capsuleFingerprint()) {
    r.error = "written for another grammar or index (fingerprint differs)";
    return r;
  }
  r.baseWords = getVarint(b, at);
  r.basePassages = getVarint(b, at);
  // The words it adds to the lexicon, front-coded, and the cell of the index each lies in.
  std::string last;
  const std::uint32_t words = getVarint(b, at);
  for (std::uint32_t w = 0; w < words && at < end; ++w) {
    const std::uint32_t shared = getVarint(b, at), rest = getVarint(b, at);
    std::string word = last.substr(0, shared);
    word.append(reinterpret_cast<const char*>(b.data() + at), std::min<std::size_t>(rest, end - at));
    at += rest;
    r.words.push_back(word);
    last = word;
  }
  {
    BitReader bits{b, at};
    for (std::uint32_t w = 0; w < words; ++w) r.cells.push_back(static_cast<double>(bits.get(kCellBits)));
  }
  const std::uint32_t count = at < end ? getVarint(b, at) : 0;
  for (std::uint32_t p = 0; p < count && at <= end; ++p) {
    std::vector<double> vocabulary, facts, codes;
    const std::uint32_t nv = getVarint(b, at);
    std::uint32_t prev = 0;
    for (std::uint32_t k = 0; k < nv; ++k) vocabulary.push_back(prev += getVarint(b, at));
    const std::uint32_t nf = getVarint(b, at);
    std::size_t slots = 0;
    for (std::uint32_t k = 0; k < nf; ++k) {
      facts.push_back(getVarint(b, at));
      slots += static_cast<std::size_t>(facts.back());
    }
    BitReader bits{b, at};
    const unsigned width = codeBits(nv);
    for (std::size_t k = 0; k < slots; ++k) codes.push_back(static_cast<double>(bits.get(width)));
    r.passages.push(Value::object().set("vocabulary", Value::Numbers(vocabulary)).set("facts", Value::Numbers(facts)).set("codes", Value::Numbers(codes)));
  }
  if (at != end) r.error = "its parts do not add up (damaged)";
  return r;
}

}  // namespace

std::uint32_t capsuleFingerprint() {
  static const std::uint32_t f = [] {
    const std::string s =
        "capsule v2: slots (fact; word point lift(u) of wordCoordinates, fnv1a-64 + fmix64, 16-bit grid of [-2,2]^3; grammar bivector, "
        "role 2.0943951, step 0.19634954); codes (vocabulary * 3 + role) * 32 + place; index cells 16^3 (spheres u_a = t, t = -2 + j/4); "
        "lexicon front-coded";
    return fnv(reinterpret_cast<const std::uint8_t*>(s.data()), s.size());
  }();
  return f;
}

CapsuleBytes packSegment(const Value& capsule, std::size_t baseWords, std::size_t basePassages) {
  CapsuleBytes out;
  std::vector<std::uint8_t>& b = out.bytes;
  b.insert(b.end(), kMagic, kMagic + 4);
  putVarint(b, kSegmentVersion);
  putU32(b, capsuleFingerprint());
  putVarint(b, static_cast<std::uint32_t>(baseWords));
  putVarint(b, static_cast<std::uint32_t>(basePassages));
  out.header = b.size();
  // The lexicon's new words, front-coded: the prefix each shares with the word before, and the rest.
  const Value& dict = capsule.at("dictionary");
  const std::size_t words = dict.size() > baseWords ? dict.size() - baseWords : 0;
  putVarint(b, static_cast<std::uint32_t>(words));
  std::string last;
  for (std::size_t w = baseWords; w < dict.size(); ++w) {
    const std::string word = dict[w].asString();
    std::size_t shared = 0;
    while (shared < last.size() && shared < word.size() && last[shared] == word[shared]) ++shared;
    putVarint(b, static_cast<std::uint32_t>(shared));
    putVarint(b, static_cast<std::uint32_t>(word.size() - shared));
    b.insert(b.end(), word.begin() + static_cast<std::ptrdiff_t>(shared), word.end());
    last = word;
  }
  out.lexicon = b.size() - out.header;
  // The index: the cell each new word lies in.
  std::size_t at = b.size();
  {
    BitWriter bits{b};
    const bool has = capsule.has("cells");
    for (std::size_t w = baseWords; w < dict.size(); ++w)
      bits.put(has && w < capsule.at("cells").size() ? static_cast<std::uint64_t>(capsule.at("cells")[w].asNumber()) : 0, kCellBits);
    bits.flush();
  }
  out.cells = b.size() - at;
  const auto& passages = capsule.at("passages").asArray();
  putVarint(b, static_cast<std::uint32_t>(passages.size() > basePassages ? passages.size() - basePassages : 0));
  for (std::size_t pi = basePassages; pi < passages.size(); ++pi) {
    const Value& p = passages[pi];
    // Level 1's codebook: the passage's vocabulary (lexicon indices, ascending, as differences).
    at = b.size();
    const Value& vocab = p.at("vocabulary");
    putVarint(b, static_cast<std::uint32_t>(vocab.size()));
    std::uint32_t prev = 0;
    for (std::size_t k = 0; k < vocab.size(); ++k) {
      const auto v = static_cast<std::uint32_t>(vocab[k].asNumber());
      putVarint(b, v - prev);
      prev = v;
    }
    out.vocabulary += b.size() - at;
    // Level 0: the facts, as run lengths of their slots.
    at = b.size();
    const Value& facts = p.at("facts");
    putVarint(b, static_cast<std::uint32_t>(facts.size()));
    for (std::size_t k = 0; k < facts.size(); ++k) putVarint(b, static_cast<std::uint32_t>(facts[k].asNumber()));
    out.facts += b.size() - at;
    // Levels 1 and 2: each slot's code, bit-packed.
    at = b.size();
    const Value& codes = p.at("codes");
    BitWriter bits{b};
    const unsigned width = codeBits(vocab.size());
    for (std::size_t k = 0; k < codes.size(); ++k) bits.put(static_cast<std::uint64_t>(codes[k].asNumber()), width);
    bits.flush();
    out.codes += b.size() - at;
  }
  putU32(b, fnv(b.data(), b.size()));
  out.header += 4;
  out.base64 = toBase64(b);
  return out;
}

CapsuleBytes packCapsule(const Value& capsule) { return packSegment(capsule, 0, 0); }

Value joinSegments(const std::vector<std::string>& segments) {
  Value dictionary = Value::array(), passages = Value::array(), report = Value::array();
  std::vector<double> cells;
  std::string refused;
  for (std::size_t i = 0; i < segments.size(); ++i) {
    Read r = readSegment(segments[i]);
    if (r.error.empty() && (r.baseWords != dictionary.size() || r.basePassages != passages.size()))
      r.error = "it extends a capsule of " + std::to_string(r.baseWords) + " words and " + std::to_string(r.basePassages) + " passages, not this one of " +
                std::to_string(dictionary.size()) + " and " + std::to_string(passages.size());
    // A segment numbers its words after those before it: once one is refused, those after it are too.
    if (r.error.empty() && !refused.empty()) r.error = "an earlier segment was refused";
    Value row = Value::object()
                    .set("bytes", static_cast<double>(r.bytes))
                    .set("words", static_cast<double>(r.words.size()))
                    .set("passages", static_cast<double>(r.passages.size()))
                    .set("used", r.error.empty());
    if (!r.error.empty()) {
      row.set("reason", r.error);
      if (refused.empty()) refused = r.error;
      report.push(row);
      continue;
    }
    for (const auto& w : r.words) dictionary.push(w);
    cells.insert(cells.end(), r.cells.begin(), r.cells.end());
    for (const auto& p : r.passages.asArray()) passages.push(p);
    report.push(row);
  }
  return Value::object()
      .set("capsule", Value::object().set("dictionary", dictionary).set("cells", Value::Numbers(cells)).set("passages", passages))
      .set("segments", report);
}

Value unpackCapsule(const std::string& base64) { return joinSegments({base64}).at("capsule"); }

const std::vector<std::string>& storedSegments() {
  static const std::vector<std::string> v = [] {
    std::vector<std::string> out(1);
    for (const char* const* line = kCapsuleBlob; *line; ++line) {
      if (**line == '\0') {  // an empty line ends a segment
        out.emplace_back();
        continue;
      }
      out.back() += *line;
    }
    if (out.back().empty()) out.pop_back();
    return out;
  }();
  return v;
}

const Value& storedCapsule() {
  static const Value v = joinSegments(storedSegments()).at("capsule");
  return v;
}

Value capsuleOf(const Value& c) {
  std::vector<std::string> given;
  if (c.has("segments"))
    for (std::size_t i = 0; i < c.at("segments").size(); ++i) given.push_back(c.at("segments")[i].asString());
  else if (c.has("packed"))
    given.push_back(c.at("packed").asString());
  else if (c.has("capsule"))
    return Value::object().set("capsule", c.at("capsule")).set("segments", Value::array()).set("stored", false);
  // Segments that extend the stored capsule (their first one starts after it) are read on top of it.
  bool stored = given.empty() || (c.has("stored") && c.at("stored").asBool());
  if (!given.empty() && !stored) {
    const Read first = readSegment(given.front());
    stored = first.error.empty() && first.baseWords > 0;
  }
  std::vector<std::string> all;
  if (stored) all = storedSegments();
  all.insert(all.end(), given.begin(), given.end());
  Value joined = joinSegments(all);
  joined.set("stored", stored);
  return joined;
}

}  // namespace syn::models::memory
