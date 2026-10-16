// Establishing learning goals and outcomes (chapter 8) in the space.
//
// Bloom's levels are points of the cognitive channel, one per level, each harder to reach than
// the one below. A written outcome is read (on the CPU: the text is input, not a model) into its
// verb, level and properties; the kernel judges it: measurable when it names an action at a
// level (not a vague verb), strong when it is also specific.
// Evidence must match the outcome: mastery climbs the hierarchy, each level's point being the
// one below lowered by a mixture (a learner cannot master a level without the levels below it);
// evidence collected at one level reads that level's point with noise, and the "met" judgement
// is a cap test at 0.6.
// Backward Design and the twin sins: a session gives minutes to content items (big ideas, the
// basic ideas they rest on, nice-to-know). Each item's mastery is a point raised by the minutes
// (1 − e^{−rate·t·η/T} as a mixture), crowded when more items are touched than a session can hold;
// a big idea's understanding needs its basics; transfer needs alignment; behaviour change is the
// cap test of a point dilated by transfer and relevance. Shared outcomes make self-assessment
// accurate. The outcome level appropriate to the audience is the highest level whose task lies
// in the audience's ZPD (dilation and caps).
#include <algorithm>
#include <array>
#include <cctype>
#include <sstream>
#include <string>

#include "syn/models/models.hpp"
#include "syn/kernel/agent.hpp"

SYN_AGENT(outcomes_agent,
  let kind = J(0u);
  if (kind < f32(0.5)) {
    let level = J(1u);
    let vague = J(2u) > f32(0.5);
    let words = J(3u);
    let specific = J(4u) > f32(0.5);
    let point = atLogit(-f32(0.7) * level, kA);
    let measurable = !vague && level >= f32(0) && words >= f32(2);
    stat(0u, f32(measurable));
    stat(1u, f32(measurable && specific));
    stat(2u, select(f32(-1), chance(point), level >= f32(0)));
  } else if (kind < f32(1.5)) {
    let items = 15u;
    let efficiency = f32(0.45) + f32(0.55) * clamp(J(16u), f32(0), f32(1));
    let alignment = clamp(J(17u), f32(0), f32(1));
    let sharedOutcomes = J(18u) > f32(0.5);
    let questions = J(19u);
    let affective = J(20u);
    var touched = f32(0);
    for (var i = 0u; i < items; i = i + 1u) { if (J(1u + i) > f32(0)) { touched = touched + f32(1); } }
    let crowding = f32(1) / (f32(1) + P(1u) * max(f32(0), touched - P(0u)));
    let relevance = f32(0.5) * alignment + f32(0.3) * questions + f32(0.2) * affective + select(f32(0), f32(0.1), sharedOutcomes);
    let judgeBias = select(f32(0.2), f32(0), sharedOutcomes);
    let judgeNoise = select(f32(0.3), f32(0.12), sharedOutcomes);
    let aptitude = exp(P(2u) * kLogisticUnit * logisticOf(U(0u)));
    ARR(mastery, V5, 15);
    for (var i = 0u; i < items; i = i + 1u) {
      let t = max(f32(0), J(1u + i) - select(f32(0), P(3u), J(1u + i) > f32(0)));
      mastery[i] = raise(atChance(f32(0), kA), f32(1) - exp(-aptitude * t * efficiency * crowding / P(12u + 3u * i)));
      stat(i, chance(mastery[i]));
    }
    var best = f32(0);
    var sumTransfer = f32(0);
    var bigs = f32(0);
    for (var j = 0u; j < items; j = j + 1u) {
      if (P(10u + 3u * j) < f32(0.5)) {
        var s = f32(0);
        var c = f32(0);
        for (var i = 0u; i < items; i = i + 1u) {
          if (P(10u + 3u * i) > f32(0.5) && P(10u + 3u * i) < f32(1.5) && u32(P(11u + 3u * i)) == j) { s = s + chance(mastery[i]); c = c + f32(1); }
        }
        let understanding = chance(mastery[j]) * select(f32(1), s / max(c, f32(1)), c > f32(0));
        let transfer = understanding * (f32(0.3) + f32(0.7) * alignment);
        sumTransfer = sumTransfer + transfer;
        best = max(best, transfer);
        bigs = bigs + f32(1);
        stat(16u, understanding);
        stat(19u, f32(2) * understanding);
      } else if (P(10u + 3u * j) < f32(1.5)) {
        stat(15u, chance(mastery[j]));
        stat(19u, chance(mastery[j]));
      } else {
        stat(17u, chance(mastery[j]));
      }
    }
    stat(18u, sumTransfer / max(bigs, f32(1)));
    let behaviour = shift(equator(kB), f32(6) * (best - f32(0.25)) + f32(2) * (relevance - f32(0.5)));
    stat(20u, f32(inside(behaviour, chanceCap(U(1u)))));
    for (var i = 0u; i < items; i = i + 1u) {
      if (J(1u + i) > f32(0)) {
        let met = inside(mastery[i], chanceCap(P(4u)));
        let judged = chance(mastery[i]) + judgeBias + judgeNoise * kLogisticUnit * logisticOf(U(2u + i)) >= P(4u);
        stat(21u, f32(met) / touched);
        stat(22u, f32(met == judged) / touched);
      }
    }
  } else if (kind < f32(2.5)) {
    let outcome = u32(J(1u));
    let evidence = u32(J(2u));
    var prev = atChance(f32(1), kA);
    ARR(x, f32, 6);
    for (var k = 0u; k < 6u; k = k + 1u) {
      let keep = chance(shift(equator(kA), P(5u) + f32(0.8) * kLogisticUnit * logisticOf(U(k)) - f32(0.45) * f32(k)));
      prev = lower(prev, f32(1) - keep);
      x[k] = chance(prev);
      stat(3u + k, x[k]);
    }
    let measured = clamp(x[evidence] + P(6u) * kLogisticUnit * logisticOf(U(10u)), f32(0), f32(1));
    stat(0u, measured);
    stat(1u, x[outcome]);
    stat(2u, f32((measured >= f32(0.6)) == (x[outcome] >= f32(0.6))));
  } else {
    let audience = J(1u);
    let scaffold = J(2u);
    var recommended = f32(-1);
    var lastIndependent = f32(-1);
    for (var k = 0u; k < 6u; k = k + 1u) {
      let task = atLogit(audience - (P(7u) + P(8u) * f32(k)), kA);
      let helped = shift(task, scaffold);
      let independent = inside(task, chanceCap(f32(0.8)));
      let proximal = !independent && inside(helped, chanceCap(f32(0.5)));
      stat(k, chance(task));
      stat(6u + k, chance(helped));
      stat(12u + k, f32(0.35) * chance(helped) * (f32(1) - chance(task)));
      stat(18u + k, select(select(f32(2), f32(1), proximal), f32(0), independent));
      if (proximal) { recommended = f32(k); }
      if (independent) { lastIndependent = f32(k); }
    }
    stat(24u, select(max(lastIndependent, f32(0)), recommended, recommended >= f32(0)));
  }
)

namespace syn::models {

namespace {

const char* kLevels[6] = {"remember", "understand", "apply", "analyze", "evaluate", "create"};
const std::vector<std::vector<std::string>>& verbs() {
  static const std::vector<std::vector<std::string>> v = {
      {"recall", "list", "state", "name", "identify", "define", "recognize", "recognise", "locate", "label", "match"},
      {"explain", "describe", "discuss", "summarize", "summarise", "paraphrase", "classify", "interpret", "illustrate", "distinguish"},
      {"apply", "implement", "use", "demonstrate", "execute", "conduct", "search", "format", "construct", "solve", "operate", "cite"},
      {"analyze", "analyse", "compare", "contrast", "examine", "differentiate", "critique", "organize", "organise", "outline", "categorize"},
      {"evaluate", "assess", "judge", "justify", "select", "choose", "defend", "appraise", "determine", "verify", "rank"},
      {"create", "design", "develop", "produce", "compose", "formulate", "plan", "propose", "build", "write"}};
  return v;
}
const std::vector<std::string> kVague = {"understand", "know", "be familiar with", "become familiar with", "learn", "appreciate",
                                         "be aware of", "become aware of", "grasp", "get to know", "be exposed to"};

struct Reading {
  std::string text, verb;
  int level = -1, actions = 0, words = 0;
  bool vague = false, specific = false;
  std::vector<std::string> issues;
};

std::string lower(std::string s) {
  for (char& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  return s;
}
std::string trim(const std::string& s) {
  const auto b = s.find_first_not_of(" \t\n,.;:-");
  if (b == std::string::npos) return {};
  const auto e = s.find_last_not_of(" \t\n,.;:");
  return s.substr(b, e - b + 1);
}
int levelOf(const std::string& w) {
  for (int k = 0; k < 6; ++k)
    for (const auto& v : verbs()[k])
      if (v == w) return k;
  return -1;
}

// Reads a written outcome (the input): stem, verb, level, actions, specificity.
Reading readOutcome(const std::string& text) {
  Reading a;
  a.text = text;
  std::string s = trim(lower(text));
  for (const char* stem : {"by the end of this session, learners will be able to", "by the end of this session learners will be able to",
                           "learners will be able to", "students will be able to", "be able to", "able to", "to"})
    if (s.compare(0, std::string(stem).size(), stem) == 0) s = trim(s.substr(std::string(stem).size()));
  for (const auto& v : kVague)
    if (s.compare(0, v.size() + 1, v + " ") == 0 || s == v) {
      a.verb = v;
      a.vague = true;
      a.level = v == "understand" ? 1 : -1;
      a.issues.push_back("\"" + v + "\" does not describe an observable action; say what learners will do (e.g. explain, apply)");
      break;
    }
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
  bool qualifier = s.find(';') != std::string::npos;
  for (std::size_t i = 1; i < words.size(); ++i)
    for (const char* q : {"for", "to", "by", "using", "of", "how", "between", "in", "from", "with", "when", "why"})
      if (words[i] == q) qualifier = true;
  a.words = static_cast<int>(words.size());
  a.specific = words.size() >= 4 && qualifier;
  if (!a.vague && a.level < 0) a.issues.push_back("no action verb from Bloom's taxonomy at the start of the outcome");
  if (a.actions > 1) a.issues.push_back("several actions: check they are assessed together, or write one outcome per action");
  if (words.size() < 2 && !a.vague) a.issues.push_back("say what knowledge the action applies to");
  else if (!a.specific && !a.vague) a.issues.push_back("too broad: add the criterion or purpose (e.g. \"for authority, reliability and accuracy\")");
  return a;
}

const std::vector<std::string> kPoor = {"Be able to search databases", "Evaluate a website", "Understand how databases work", "Understand primary sources"};
const std::vector<std::string> kStrong = {"Apply Boolean search strategies effectively to broaden and narrow searches.",
                                          "Evaluate websites for authority, reliability, and accuracy; identify trustworthy sites.",
                                          "Describe how information is parsed and stored in a database.",
                                          "Define primary, secondary, and tertiary sources; identify examples of each."};

std::vector<std::string> textsOf(const Value& c) {
  std::vector<std::string> t;
  const Value& v = c.at("texts");
  if (v.isArray())
    for (const auto& x : v.asArray())
      if (x.isString() && t.size() < 50) t.push_back(x.asString());
  return t;
}

// Content of a one-shot library session (Figure 8.3, Tables 8.5–8.6).
struct Item {
  const char* name;
  int kind;      // 0 big idea, 1 basic idea, 2 nice to know
  int bigIdea;   // for basic ideas
  int level;     // Bloom level
  double minutes;
};
const Item kItems[15] = {{"searching as strategic exploration", 0, -1, 3, 10}, {"Boolean operators", 1, 0, 2, 8}, {"nesting and truncation", 1, 0, 2, 7},
                         {"credible sources", 0, -1, 4, 10}, {"authority", 1, 3, 4, 8}, {"currency and accuracy", 1, 3, 4, 7},
                         {"organization of information", 0, -1, 1, 9}, {"keyword vs subject searching", 1, 6, 1, 7},
                         {"classification and call numbers", 1, 6, 1, 6}, {"catalogue limiters", 1, 6, 2, 6},
                         {"library website tour", 2, -1, 0, 6}, {"interlibrary loan form", 2, -1, 0, 4}, {"chat reference link", 2, -1, 0, 2},
                         {"Melvil Dewey", 2, -1, 0, 2}, {"the term OPAC", 2, -1, 0, 2}};
const char* kKinds[3] = {"big idea", "basic idea", "worth being familiar with"};

std::vector<int> priority() {
  std::vector<int> order;
  for (int j = 0; j < 15; ++j) {
    if (kItems[j].kind != 0) continue;
    for (int i = 0; i < 15; ++i)
      if (kItems[i].kind == 1 && kItems[i].bigIdea == j) order.push_back(i);
    order.push_back(j);
  }
  for (int i = 0; i < 15; ++i)
    if (std::find(order.begin(), order.end(), i) == order.end()) order.push_back(i);
  return order;
}

std::vector<double> proportional(const std::vector<int>& chosen, double minutes) {
  std::vector<double> t(15, 0.0);
  double need = 0.0;
  for (int i : chosen) need += kItems[i].minutes;
  for (int i : chosen) t[i] = minutes * kItems[i].minutes / need;
  return t;
}

// Session rows: kind 1, minutes per item (15), active share, alignment, shared, questions, affective.
std::vector<double> sessionRow(const std::vector<double>& minutes, double active, double alignment, bool shared, double questions, double affective) {
  std::vector<double> r = {1};
  r.insert(r.end(), minutes.begin(), minutes.end());
  r.insert(r.end(), {active, alignment, shared ? 1.0 : 0.0, questions, affective});
  return r;
}
std::vector<double> coverage(double minutes) { return sessionRow(std::vector<double>(15, minutes / 15), 0.05, 0.3, false, 0, 0); }
std::vector<double> activity(double minutes) {
  std::vector<int> chosen;
  for (int i = 0; i < 15; ++i)
    if (kItems[i].kind == 1 && kItems[i].level == 2) chosen.push_back(i);
  return sessionRow(proportional(chosen, minutes), 0.85, 0.15, false, 0, 0);
}
std::vector<double> backward(double minutes, int outcomes, bool shared = true, double questions = 0.7, double affective = 0.5) {
  const auto order = priority();
  const std::vector<int> chosen(order.begin(), order.begin() + std::min<std::size_t>(15, static_cast<std::size_t>(outcomes)));
  return sessionRow(proportional(chosen, minutes), 0.6, 0.9, shared, questions, affective);
}

std::vector<double> pad(std::vector<double> r) {
  r.resize(21, 0.0);
  return r;
}

}  // namespace

const kernel::Model* outcomesModel() {
  static const kernel::Model model = [] {
    kernel::Model m;
    m.name = "outcomes";
    m.title = "Learning outcomes: Bloom's levels as points, evidence against levels, the twin sins and Backward Design";
    m.maxStats = 25;
    SYN_USE_AGENT(m, outcomes_agent);
    m.compile = [](const Value& spec) {
      const Value& c = spec.sub("outcomes");
      const double minutes = c.num("minutes", 50.0);
      const int count = std::max(1, c.integer("outcomes", 4));
      // 0 capacity, 1 crowding, 2 aptitude spread, 3 transition minutes, 4 "met" threshold, 5 depth of teaching,
      // 6 noise of evidence, 7–8 Bloom difficulty base and step; items from 10: kind, big idea, minutes needed.
      std::vector<double> params = {4.0, 0.15, 0.4, 1.0, 0.6, c.num("depth", 1.5), 0.1, -1.5, 0.7, 0.0};
      for (const auto& it : kItems) params.insert(params.end(), {double(it.kind), double(it.bigIdea), it.minutes});
      std::vector<std::vector<double>> texts;
      std::vector<std::string> all = kPoor;
      all.insert(all.end(), kStrong.begin(), kStrong.end());
      for (const auto& t : textsOf(c)) all.push_back(t);
      for (const auto& t : all) {
        const Reading a = readOutcome(t);
        texts.push_back(pad({0, double(a.level), a.vague ? 1.0 : 0.0, double(a.words), a.specific ? 1.0 : 0.0}));
      }
      std::vector<std::vector<double>> sessions = {coverage(minutes), activity(minutes), backward(minutes, count)};
      for (double len : {30.0, 60.0, 90.0})
        for (int k = 1; k <= 12; ++k) sessions.push_back(backward(len, k));
      for (int f = 0; f < 4; ++f) sessions.push_back(backward(minutes, count, f >= 1, f >= 2 ? 0.7 : 0.0, f >= 3 ? 0.5 : 0.0));
      std::vector<std::vector<double>> evidence, levels;
      for (int e = 0; e < 6; ++e) evidence.push_back(pad({2, 2, double(e)}));
      for (double th : {-2.0, -1.0, 0.0, 1.0, 2.0}) levels.push_back(pad({3, th, c.num("scaffold", 1.5)}));
      const std::uint32_t learners = static_cast<std::uint32_t>(c.integer("learners", 400));
      return std::vector<Plan>{makePlan("outcomes", 1, 3, params, texts, seedOf(spec)), makePlan("outcomes", learners, 23, params, sessions, seedOf(spec)),
                               makePlan("outcomes", learners, 9, params, evidence, seedOf(spec)), makePlan("outcomes", 1, 25, params, levels, seedOf(spec))};
    };
    m.summarise = [](const Value& spec, const std::vector<Plan>& plans, const std::vector<Output>& out) {
      const Value& c = spec.sub("outcomes");
      Value r = Value::object();
      // Stage 1: written outcomes.
      std::vector<std::string> all = kPoor;
      all.insert(all.end(), kStrong.begin(), kStrong.end());
      for (const auto& t : textsOf(c)) all.push_back(t);
      auto analysis = [&](std::uint32_t j) {
        const Reading a = readOutcome(all[j]);
        const bool measurable = out[0].mean(plans[0], j, 0) > 0.5, strong = out[0].mean(plans[0], j, 1) > 0.5;
        return Value::object()
            .set("text", a.text)
            .set("verb", a.verb)
            .set("level", a.level)
            .set("levelName", a.level >= 0 ? kLevels[a.level] : "none")
            .set("levelChance", out[0].mean(plans[0], j, 2))
            .set("vague", a.vague)
            .set("measurable", measurable)
            .set("specific", a.specific)
            .set("strong", strong)
            .set("actions", a.actions)
            .set("issues", a.issues);
      };
      Value table = Value::array();
      for (std::uint32_t i = 0; i < 4; ++i) table.push(Value::object().set("poor", analysis(i)).set("strong", analysis(4 + i)));
      r.set("table83", table);
      Value own = Value::array();
      for (std::uint32_t j = 8; j < all.size(); ++j) own.push(analysis(j));
      r.set("analysed", own);
      r.set("levels", std::vector<std::string>(kLevels, kLevels + 6));
      std::vector<std::string> verbList;
      for (const auto& v : verbs()) {
        std::string s;
        for (const auto& w : v) s += (s.empty() ? "" : ", ") + w;
        verbList.push_back(s);
      }
      r.set("verbs", verbList);
      // Stage 3 vs the twin sins.
      Value content = Value::array();
      for (const auto& it : kItems)
        content.push(Value::object().set("name", it.name).set("kind", kKinds[it.kind]).set("bigIdea", it.bigIdea >= 0 ? kItems[it.bigIdea].name : "")
                         .set("level", kLevels[it.level]).set("minutes", it.minutes));
      r.set("content", content);
      const Plan& s = plans[1];
      const Output& os = out[1];
      const int count = std::max(1, c.integer("outcomes", 4));
      auto session = [&](std::uint32_t j, const std::string& name) {
        std::vector<double> minutes(s.jobRows.begin() + j * s.jobStride + 1, s.jobRows.begin() + j * s.jobStride + 16);
        double touched = 0;
        for (double t : minutes) touched += t > 0 ? 1 : 0;
        return Value::object()
            .set("plan", name)
            .set("mastery", Value::Numbers(os.range(s, j, 0, 15)))
            .set("minutes", Value::Numbers(minutes))
            .set("itemsTouched", touched)
            .set("coverage", touched / 15)
            .set("basics", os.mean(s, j, 15) / 7)
            .set("bigIdeas", os.mean(s, j, 16) / 3)
            .set("niceToKnow", os.mean(s, j, 17) / 5)
            .set("enduring", os.mean(s, j, 19))
            .set("outcomesMet", os.mean(s, j, 21))
            .set("transfer", os.mean(s, j, 18))
            .set("behaviourChange", os.mean(s, j, 20))
            .set("selfAssessment", os.mean(s, j, 22));
      };
      Value planList = Value::array();
      planList.push(session(0, "coverage-based")).push(session(1, "activity-based")).push(session(2, "backward design (" + std::to_string(count) + " outcomes)"));
      r.set("plans", planList);
      std::vector<double> ks;
      for (int k = 1; k <= 12; ++k) ks.push_back(k);
      r.set("outcomeCounts", Value::Numbers(ks));
      Value sweep = Value::array();
      std::uint32_t j = 3;
      for (double len : {30.0, 60.0, 90.0}) {
        std::vector<double> enduring, met;
        for (int k = 1; k <= 12; ++k, ++j) {
          enduring.push_back(os.mean(s, j, 19));
          met.push_back(os.mean(s, j, 21));
        }
        sweep.push(Value::object().set("minutes", len).set("enduring", Value::Numbers(enduring)).set("met", Value::Numbers(met)));
      }
      r.set("sweep", sweep);
      Value framing = Value::array();
      const char* names[] = {"outcomes not shared", "+ shared outcomes", "+ essential questions", "+ affective framing"};
      for (int f = 0; f < 4; ++f, ++j) framing.push(Value::object().set("name", names[f]).set("behaviourChange", os.mean(s, j, 20)).set("selfAssessment", os.mean(s, j, 22)));
      r.set("framing", framing);
      // Stage 2: evidence at each level for an "apply" outcome.
      const Plan& e = plans[2];
      std::vector<double> bias, accuracy;
      for (std::uint32_t k = 0; k < 6; ++k) {
        bias.push_back(out[2].mean(e, k, 0) - out[2].mean(e, k, 1));
        accuracy.push_back(out[2].mean(e, k, 2));
      }
      r.set("levelMastery", Value::Numbers(out[2].range(e, 0, 3, 6))).set("evidenceBias", Value::Numbers(bias)).set("evidenceAccuracy", Value::Numbers(accuracy));
      // The level appropriate to the audience.
      const Plan& f = plans[3];
      Value fits = Value::array();
      const double audiences[] = {-2.0, -1.0, 0.0, 1.0, 2.0};
      for (std::uint32_t a = 0; a < 5; ++a)
        fits.push(Value::object()
                      .set("audience", audiences[a])
                      .set("alone", Value::Numbers(out[3].range(f, a, 0, 6)))
                      .set("assisted", Value::Numbers(out[3].range(f, a, 6, 6)))
                      .set("gain", Value::Numbers(out[3].range(f, a, 12, 6)))
                      .set("zone", Value::Numbers(out[3].range(f, a, 18, 6)))
                      .set("recommended", std::lround(out[3].mean(f, a, 24))));
      r.set("levelFit", fits);
      return r;
    };
    return m;
  }();
  return &model;
}

}  // namespace syn::models
