#include "syn/outcomes/bloom.hpp"

#include <algorithm>
#include <cctype>
#include <sstream>
#include <stdexcept>

#include "syn/core/math.hpp"
#include "syn/core/random.hpp"

namespace syn {

const char* bloomName(BloomLevel l) {
  static const char* names[kBloomLevels] = {"remember", "understand", "apply", "analyze", "evaluate", "create"};
  const int k = static_cast<int>(l);
  return k >= 0 && k < kBloomLevels ? names[k] : "?";
}

bool higherOrder(BloomLevel l) { return static_cast<int>(l) >= static_cast<int>(BloomLevel::Analyze); }

const std::vector<std::string>& bloomVerbs(BloomLevel l) {
  static const std::array<std::vector<std::string>, kBloomLevels> verbs = {{
      {"recall", "list", "state", "name", "identify", "define", "recognize", "recognise", "locate", "label", "match"},
      {"explain", "describe", "discuss", "summarize", "summarise", "paraphrase", "classify", "interpret", "illustrate", "distinguish"},
      {"apply", "implement", "use", "demonstrate", "execute", "conduct", "search", "format", "construct", "solve", "operate", "cite"},
      {"analyze", "analyse", "compare", "contrast", "examine", "differentiate", "critique", "organize", "organise", "outline", "categorize"},
      {"evaluate", "assess", "judge", "justify", "select", "choose", "defend", "appraise", "determine", "verify", "rank"},
      {"create", "design", "develop", "produce", "compose", "formulate", "plan", "propose", "build", "write"},
  }};
  return verbs[static_cast<int>(l)];
}

const std::vector<std::string>& vagueVerbs() {
  static const std::vector<std::string> v = {"understand", "know", "be familiar with", "become familiar with", "learn", "appreciate",
                                             "be aware of", "become aware of", "grasp", "get to know", "be exposed to"};
  return v;
}

namespace {

std::string lower(std::string s) {
  for (char& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  return s;
}

bool startsWith(const std::string& s, const std::string& p) { return s.compare(0, p.size(), p) == 0; }

std::string trim(std::string s) {
  const auto b = s.find_first_not_of(" \t\n,.;:-");
  if (b == std::string::npos) return {};
  const auto e = s.find_last_not_of(" \t\n,.;:");
  return s.substr(b, e - b + 1);
}

int levelOf(const std::string& word) {
  for (int k = 0; k < kBloomLevels; ++k)
    for (const auto& v : bloomVerbs(static_cast<BloomLevel>(k)))
      if (v == word) return k;
  return -1;
}

}  // namespace

OutcomeAnalysis analyseOutcome(const std::string& text) {
  OutcomeAnalysis a;
  a.text = text;
  std::string s = trim(lower(text));
  // Strip the usual stems.
  for (const char* stem : {"by the end of this session, learners will be able to", "by the end of this session learners will be able to",
                           "learners will be able to", "students will be able to", "be able to", "able to", "to"})
    if (startsWith(s, stem)) s = trim(s.substr(std::string(stem).size()));
  for (const auto& v : vagueVerbs())
    if (startsWith(s, v + " ") || s == v) {
      a.verb = v;
      a.vague = true;
      a.level = v == "understand" ? static_cast<int>(BloomLevel::Understand) : -1;
      a.issues.push_back("\"" + v + "\" does not describe an observable action; say what learners will do (e.g. explain, apply)");
      break;
    }
  // Count action verbs: the leading word and any after "and", "or", "then", ";" or ",".
  std::istringstream in(s);
  std::vector<std::string> words;
  std::vector<bool> afterBreak;
  bool brk = true;
  for (std::string raw; in >> raw;) {
    const std::string w = trim(raw);
    afterBreak.push_back(brk);
    words.push_back(w);
    brk = w == "and" || w == "or" || w == "then" || raw.back() == ';' || raw.back() == ',';
  }
  for (std::size_t i = 0; i < words.size(); ++i) {
    const int k = levelOf(words[i]);
    if (k >= 0 && afterBreak[i]) {
      ++a.actions;
      if (a.verb.empty()) {
        a.verb = words[i];
        a.level = k;
      }
    }
  }
  // Specific: the knowledge the action applies to is named with a criterion or condition, not
  // just an object ("search databases", "evaluate a website" are too broad to assess).
  bool qualifier = s.find(';') != std::string::npos;
  for (std::size_t i = 1; i < words.size(); ++i)
    for (const char* q : {"for", "to", "by", "using", "of", "how", "between", "in", "from", "with", "when", "why"})
      if (words[i] == q) qualifier = true;
  a.specific = words.size() >= 4 && qualifier;
  if (!a.vague && a.level < 0) a.issues.push_back("no action verb from Bloom's taxonomy at the start of the outcome");
  if (a.actions > 1) a.issues.push_back("several actions: check they are assessed together, or write one outcome per action");
  if (words.size() < 2 && !a.vague) a.issues.push_back("say what knowledge the action applies to");
  else if (!a.specific && !a.vague) a.issues.push_back("too broad: add the criterion or purpose (e.g. \"for authority, reliability and accuracy\")");
  a.measurable = !a.vague && a.level >= 0 && words.size() >= 2;
  a.strong = a.measurable && a.specific;
  return a;
}

LevelMastery simulateLevelMastery(int learners, double depth, std::uint64_t seed) {
  if (learners < 1) throw std::invalid_argument("simulateLevelMastery: need learners");
  LevelMastery m;
  for (int i = 0; i < learners; ++i) {
    Rng rng = Rng::stream(seed, static_cast<std::uint64_t>(i));
    std::array<double, kBloomLevels> x{};
    double prev = 1.0;
    for (int k = 0; k < kBloomLevels; ++k) {
      // Each step up keeps a share of the level below; depth (teaching aimed higher) raises it.
      const double keep = clamp01(logistic(depth + 0.8 * rng.normal() - 0.45 * k));
      x[k] = prev * keep;
      prev = x[k];
    }
    m.learners.push_back(x);
  }
  return m;
}

EvidenceCheck checkEvidence(const LevelMastery& m, BloomLevel outcome, BloomLevel evidence, double noise, std::uint64_t seed) {
  EvidenceCheck c{outcome, evidence};
  const int o = static_cast<int>(outcome), e = static_cast<int>(evidence);
  Rng rng(seed);
  const double n = static_cast<double>(m.learners.size());
  for (const auto& x : m.learners) {
    const double measured = clamp01(x[e] + noise * rng.normal());
    c.measured += measured / n;
    c.actual += x[o] / n;
    c.accuracy += ((measured >= 0.6) == (x[o] >= 0.6) ? 1.0 : 0.0) / n;
  }
  c.bias = c.measured - c.actual;
  return c;
}

}  // namespace syn
