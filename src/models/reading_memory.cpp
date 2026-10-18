// What the reader keeps between runs: its memory of reading, note-taking and constructing habits,
// and its knowledge, a capsule of what it read as points of the space (see reading_memory.hpp).
#include "reading_memory.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstring>
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
const char* const kConstructHabits[kHabitCount] = {"overlap of the fact with the question (BM25)", "says two facts (this one and the next)",
                                                   "says the whole fact", "says the predicate (relation, object)", "says the object",
                                                   "says the subject", "holds what is asked (a number, a name)", "covers the question"};
const char* const kRoleNames[kRoles] = {"subject", "relation", "object"};
const double kHabitPrior[kHabitCount] = {1, 0, 0, 0, 0, -1, 0, 0};
const double kNotePrior[kHabitCount] = {1, 0, 0, 0, 0, 0, 0, 0};
const double kConstructPrior[kHabitCount] = {1, 0, 1, 0, 0, 0, 0, 0};

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
  f += " construct v1:";
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

Memory recall(bool fresh, const std::string& fingerprintOf) {
  Memory m;
  std::copy(kHabitPrior, kHabitPrior + kHabitCount, m.habits);
  std::copy(kNotePrior, kNotePrior + kHabitCount, m.notes);
  std::copy(kConstructPrior, kConstructPrior + kHabitCount, m.construct);
  if (fresh) {
    m.reason = "started fresh";
    return m;
  }
  const std::string stored = fingerprintOf.empty() ? kStored.fingerprint : fingerprintOf;
  if (stored != fingerprint()) {
    m.reason = "the memory was learned for other habits (fingerprint differs); started from the prior";
    return m;
  }
  m.used = true;
  m.generation = kStored.generation;
  m.lessons = kStored.lessons;
  m.design = kStored.design;
  m.source = kStored.source;
  m.test = kStored.test;
  m.notesDesign = kStored.notesDesign;
  m.notesTest = kStored.notesTest;
  std::copy(kStored.habits, kStored.habits + kHabitCount, m.habits);
  std::copy(kStored.notes, kStored.notes + kHabitCount, m.notes);
  m.constructDesign = kStored.constructDesign;
  m.constructTest = kStored.constructTest;
  std::copy(kStored.construct, kStored.construct + kHabitCount, m.construct);
  return m;
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

Fact factOf(const std::vector<std::string>& t) {
  // The verb: the earliest auxiliary, or word with a verb's ending (-s, -ed) that does not follow a
  // preposition or determiner (as "of flowers" does); else, after a plural first word, the next
  // word ("Honeybees communicate").
  static const std::set<std::string> before = {"of", "with", "in", "on", "for", "by", "to", "from", "other", "many", "some", "all",
                                               "these", "those", "their", "its", "his", "her", "our", "your", "and", "or"};
  std::size_t verb = t.size();
  for (std::size_t i = 0; i < t.size(); ++i) {
    const std::string l = lower(t[i]);
    const bool ending = i > 0 && ((l.size() > 3 && l.compare(l.size() - 2, 2, "ed") == 0) || (l.size() > 3 && l.back() == 's' && l[l.size() - 2] != 's')) &&
                        !text::stopwords().count(l) && !std::isupper(static_cast<unsigned char>(t[i][0])) && !before.count(lower(t[i - 1]));
    if (auxiliaries().count(l) || ending) {
      verb = i;
      break;
    }
  }
  if (verb == t.size() && t.size() > 2) {
    const std::string first = lower(t[0]), second = lower(t[1]);
    if (first.size() > 3 && first.back() == 's' && !text::stopwords().count(second) && !std::isupper(static_cast<unsigned char>(t[1][0]))) verb = 1;
  }
  Fact f;
  if (verb == t.size()) {
    f.object.assign(t.begin(), t.begin() + static_cast<std::ptrdiff_t>(std::min<std::size_t>(t.size(), 24)));
    return f;
  }
  std::size_t end = verb + 1;
  while (end < t.size() && end < verb + 4 && followsVerb(t[end])) ++end;
  f.subject.assign(t.begin() + static_cast<std::ptrdiff_t>(verb > 8 ? verb - 8 : 0), t.begin() + static_cast<std::ptrdiff_t>(verb));
  f.relation.assign(t.begin() + static_cast<std::ptrdiff_t>(verb), t.begin() + static_cast<std::ptrdiff_t>(end));
  f.object.assign(t.begin() + static_cast<std::ptrdiff_t>(end), t.begin() + static_cast<std::ptrdiff_t>(std::min(t.size(), end + 16)));
  return f;
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

std::vector<Fact> grammar(const std::string& passage) {
  std::vector<Fact> out;
  for (const auto& sentence : sentences(passage))
    for (const auto& clause : clausesOf(sentence))
      if (clause.size() >= 3) out.push_back(factOf(clause));
  return out;
}

void wordCoordinates(const std::string& word, float u[3]) {
  std::uint32_t h = 2166136261u;  // FNV-1a
  for (unsigned char c : word) h = (h ^ c) * 16777619u;
  for (int k = 0; k < 3; ++k) {
    h = h * 1664525u + 1013904223u;
    u[k] = static_cast<float>((h >> 8) & 0xffffu) / 65535.0f * 4.0f - 2.0f;
  }
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

std::vector<Candidates> constructCandidates(const std::vector<Item>& items, const std::vector<std::vector<Fact>>& facts) {
  // How informative each word is, learned from the facts it recalls (each fact a document).
  std::map<std::string, double> df;
  double n = 0, totalLen = 0;
  for (const auto& fs : facts)
    for (const auto& f : fs) {
      const auto ws = words(joined(wordsOf(f)), true);
      for (const auto& w : std::set<std::string>(ws.begin(), ws.end())) df[w] += 1;
      totalLen += static_cast<double>(ws.size());
      n += 1;
    }
  const double avg = totalLen / std::max(1.0, n);
  auto idf = [&](const std::string& w) {
    auto it = df.find(w);
    const double d = it == df.end() ? 0.0 : it->second;
    return std::max(0.0, std::log((n - d + 0.5) / (d + 0.5) + 1.0));
  };
  std::vector<Candidates> out;
  for (std::size_t i = 0; i < items.size(); ++i) {
    const auto& q = items[i].question;
    std::set<std::string> qw;
    for (const auto& w : words(q, true)) qw.insert(w);
    const std::string lq = lower(q);
    const bool asksNumber = lq.find("how many") != std::string::npos || lq.find("how much") != std::string::npos ||
                            lq.find("percent") != std::string::npos || lq.find("what year") != std::string::npos || lq.rfind("when", 0) == 0 ||
                            lq.find("how long") != std::string::npos || lq.find("how old") != std::string::npos;
    const bool asksName = lq.rfind("who", 0) == 0 || lq.find(" who ") != std::string::npos || lq.find("which person") != std::string::npos;
    const auto& fs = facts[i];
    const std::size_t nf = std::min<std::size_t>(fs.size(), 24);
    std::vector<double> bm(nf);
    double maxBm = 1e-9;
    for (std::size_t k = 0; k < nf; ++k) {
      const auto ws = words(joined(wordsOf(fs[k])), true);
      std::map<std::string, double> tf;
      for (const auto& w : ws) tf[w] += 1;
      const double norm = 1.2 * (1 - 0.75 + 0.75 * static_cast<double>(ws.size()) / avg);
      for (const auto& w : qw)
        if (tf.count(w)) bm[k] += idf(w) * tf[w] * 2.2 / (tf[w] + norm);
      maxBm = std::max(maxBm, bm[k]);
    }
    Candidates p;
    for (std::size_t k = 0; k < nf; ++k) {
      const Fact& f = fs[k];
      std::vector<std::string> predicate = f.relation;
      predicate.insert(predicate.end(), f.object.begin(), f.object.end());
      const std::vector<std::string>* said[4] = {nullptr, &predicate, &f.object, &f.subject};
      const std::vector<std::string> whole = wordsOf(f);
      said[0] = &whole;
      // The grammar's ways of saying it: the whole fact, its predicate, its object, its subject, and
      // the whole fact with the next one.
      std::vector<std::string> two = whole;
      if (k + 1 < nf) {
        const auto after = wordsOf(fs[k + 1]);
        two.push_back(".");
        two.insert(two.end(), after.begin(), after.end());
      }
      for (int pattern = 0; pattern < 5; ++pattern) {
        if (pattern == 4 && k + 1 >= nf) continue;
        const std::vector<std::string>& say = pattern == 4 ? two : *said[pattern];
        if (say.empty() || (pattern == 1 && f.relation.empty())) continue;
        const std::string t = realize(say);
        const auto ws = words(t, true);
        std::set<std::string> have(ws.begin(), ws.end());
        double covered = 0;
        for (const auto& w : qw) covered += have.count(w) ? 1.0 : 0.0;
        const double fit = (asksNumber && hasDigit(t)) || (asksName && hasName(t)) ? 1.0 : 0.0;
        p.texts.push_back(t);
        p.pair.push_back(false);
        p.habits.push_back({bm[k] / maxBm, pattern == 4 ? 1.0 : 0.0, pattern == 0 || pattern == 4 ? 1.0 : 0.0, pattern == 1 ? 1.0 : 0.0,
                            pattern == 2 ? 1.0 : 0.0, pattern == 3 ? 1.0 : 0.0, fit, qw.empty() ? 0.0 : covered / static_cast<double>(qw.size())});
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

CapsuleBytes packCapsule(const Value& capsule) {
  CapsuleBytes out;
  std::vector<std::uint8_t>& b = out.bytes;
  // The lexicon, front-coded: the prefix it shares with the word before, and the rest.
  const Value& dict = capsule.at("dictionary");
  putVarint(b, static_cast<std::uint32_t>(dict.size()));
  std::string last;
  for (std::size_t w = 0; w < dict.size(); ++w) {
    const std::string word = dict[w].asString();
    std::size_t shared = 0;
    while (shared < last.size() && shared < word.size() && last[shared] == word[shared]) ++shared;
    putVarint(b, static_cast<std::uint32_t>(shared));
    putVarint(b, static_cast<std::uint32_t>(word.size() - shared));
    b.insert(b.end(), word.begin() + static_cast<std::ptrdiff_t>(shared), word.end());
    last = word;
  }
  out.lexicon = b.size();
  const auto& passages = capsule.at("passages").asArray();
  putVarint(b, static_cast<std::uint32_t>(passages.size()));
  for (const auto& p : passages) {
    // Level 1's codebook: the passage's vocabulary (lexicon indices, ascending, as differences).
    std::size_t at = b.size();
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
    const unsigned bits = codeBits(vocab.size());
    std::uint64_t acc = 0;
    unsigned have = 0;
    for (std::size_t k = 0; k < codes.size(); ++k) {
      acc |= static_cast<std::uint64_t>(codes[k].asNumber()) << have;
      have += bits;
      while (have >= 8) {
        b.push_back(static_cast<std::uint8_t>(acc & 0xff));
        acc >>= 8;
        have -= 8;
      }
    }
    if (have > 0) b.push_back(static_cast<std::uint8_t>(acc & 0xff));
    out.codes += b.size() - at;
  }
  for (std::size_t i = 0; i < b.size(); i += 3) {
    const std::uint32_t n = (std::uint32_t{b[i]} << 16) | (i + 1 < b.size() ? std::uint32_t{b[i + 1]} << 8 : 0u) | (i + 2 < b.size() ? b[i + 2] : 0u);
    out.base64 += kBase64[(n >> 18) & 63];
    out.base64 += kBase64[(n >> 12) & 63];
    out.base64 += i + 1 < b.size() ? kBase64[(n >> 6) & 63] : '=';
    out.base64 += i + 2 < b.size() ? kBase64[n & 63] : '=';
  }
  return out;
}

Value unpackCapsule(const std::string& base64) {
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
  std::size_t at = 0;
  Value dictionary = Value::array(), passages = Value::array();
  std::string last;
  const std::uint32_t words = getVarint(b, at);
  for (std::uint32_t w = 0; w < words; ++w) {
    const std::uint32_t shared = getVarint(b, at), rest = getVarint(b, at);
    std::string word = last.substr(0, shared);
    word.append(reinterpret_cast<const char*>(b.data() + at), std::min<std::size_t>(rest, b.size() - at));
    at += rest;
    dictionary.push(word);
    last = word;
  }
  const std::uint32_t count = at < b.size() ? getVarint(b, at) : 0;
  for (std::uint32_t p = 0; p < count; ++p) {
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
    const unsigned bits = codeBits(nv);
    std::uint64_t bitsAcc = 0;
    unsigned inAcc = 0;
    for (std::size_t k = 0; k < slots; ++k) {
      while (inAcc < bits && at < b.size()) {
        bitsAcc |= static_cast<std::uint64_t>(b[at++]) << inAcc;
        inAcc += 8;
      }
      codes.push_back(static_cast<double>(bitsAcc & ((std::uint64_t{1} << bits) - 1)));
      bitsAcc >>= bits;
      inAcc -= bits;
    }
    passages.push(Value::object().set("vocabulary", Value::Numbers(vocabulary)).set("facts", Value::Numbers(facts)).set("codes", Value::Numbers(codes)));
  }
  return Value::object().set("dictionary", dictionary).set("passages", passages);
}

const Value& storedCapsule() {
  static const Value v = [] {
    std::string b64;
    for (const char* const* line = kCapsuleBlob; *line; ++line) b64 += *line;
    return unpackCapsule(b64);
  }();
  return v;
}

}  // namespace syn::models::memory
