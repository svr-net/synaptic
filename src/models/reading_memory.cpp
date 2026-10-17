// What the reader keeps between runs: its memory of reading and note-taking habits, and the notes
// it took on what it read (see reading_memory.hpp).
#include "reading_memory.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
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
const double kHabitPrior[kHabitCount] = {1, 0, 0, 0, 0, -1, 0, 0};
const double kNotePrior[kHabitCount] = {1, 0, 0, 0, 0, 0, 0, 0};

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
};
const Stored kStored =
#include "student_memory.inc"
    ;

// The stored knowledge: one entry per passage read (written by tools/qa_golden).
struct StoredNote {
  const char* source;
  int generation;
  int perPassage;
  int passageChars;
  const char* notes;
};
const StoredNote kStoredNotes[] = {
#include "reader_knowledge.inc"
    {nullptr, 0, 0, 0, nullptr}};

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
  return m;
}

const Knowledge& knowledge() {
  static const Knowledge k = [] {
    Knowledge out;
    double keptChars = 0, allChars = 0;
    for (const StoredNote* n = kStoredNotes; n->source; ++n) {
      out.sources.push_back(n->source);
      out.notes.push_back(n->notes);
      out.perPassage = n->perPassage;
      out.generation = n->generation;
      keptChars += static_cast<double>(std::strlen(n->notes));
      allChars += n->passageChars;
    }
    out.kept = allChars > 0 ? keptChars / allChars : 0.0;
    return out;
  }();
  return k;
}

}  // namespace syn::models::memory
