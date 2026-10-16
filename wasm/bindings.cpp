// Embind wrapper exposing the library to JavaScript.
//
// Every entry point takes a plain JS "spec" object (population, course, instructional design,
// simulation settings and the page's own experiment settings) and returns a plain JS object of
// numbers, strings and Float64Arrays. Errors are returned as { error: "message" } rather than
// thrown across the boundary.

#include <emscripten/bind.h>
#include <emscripten/val.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <string>
#include <vector>

#include "syn/syn.hpp"

using emscripten::val;
using namespace syn;

namespace {

// ------------------------------------------------------------------ JS conversion helpers

bool has(const val& o, const char* key) {
  if (o.isUndefined() || o.isNull()) return false;
  const val v = o[key];
  return !(v.isUndefined() || v.isNull());
}

double num(const val& o, const char* key, double fallback) { return has(o, key) ? o[key].as<double>() : fallback; }
int integer(const val& o, const char* key, int fallback) { return static_cast<int>(std::lround(num(o, key, fallback))); }
bool flag(const val& o, const char* key, bool fallback) { return has(o, key) ? o[key].as<bool>() : fallback; }
std::string str(const val& o, const char* key, const std::string& fallback) {
  return has(o, key) ? o[key].as<std::string>() : fallback;
}
val sub(const val& o, const char* key) { return has(o, key) ? o[key] : val::object(); }

std::vector<double> vec(const val& v) { return emscripten::vecFromJSArray<double>(v); }

val arr(const std::vector<double>& v) {
  return val(emscripten::typed_memory_view(v.size(), v.data())).call<val>("slice");
}

template <std::size_t N>
val arr(const std::array<double, N>& a) {
  return arr(std::vector<double>(a.begin(), a.end()));
}

template <class T>
val typedArray(const std::vector<T>& v) {
  return val(emscripten::typed_memory_view(v.size(), v.data())).call<val>("slice");
}

val arrOfArr(const std::vector<std::vector<double>>& rows) {
  val out = val::array();
  for (const auto& r : rows) out.call<void>("push", arr(r));
  return out;
}

val strings(const std::vector<std::string>& v) {
  val out = val::array();
  for (const auto& s : v) out.call<void>("push", s);
  return out;
}

double nowMs() {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

template <class F>
val guarded(F&& body) {
  try {
    return body();
  } catch (const std::exception& e) {
    val o = val::object();
    o.set("error", std::string(e.what()));
    return o;
  } catch (...) {
    val o = val::object();
    o.set("error", std::string("unknown C++ exception"));
    return o;
  }
}

std::uint64_t seedOf(const val& spec) { return static_cast<std::uint64_t>(num(sub(spec, "sim"), "seed", 2024)); }

// ------------------------------------------------------------------ shared spec parsing

const char* const kLeverKeys[kLevers] = {"reinforcement", "teacherDirection", "empathy",      "choice",
                                         "relevance",     "transparency",     "priorActivation", "retrievalPractice",
                                         "spacing",       "interleaving",     "feedback",     "problemFirst",
                                         "scaffolding",   "peerDialogue",     "inclusivity",  "growthFeedback"};

InstructionalDesign parseDesign(const val& d) {
  InstructionalDesign out;
  for (int k = 0; k < kLevers; ++k) out.lever[k] = num(d, kLeverKeys[k], 0.0);
  return out.clamped();
}

val designToJs(const InstructionalDesign& d) {
  val o = val::object();
  for (int k = 0; k < kLevers; ++k) o.set(kLeverKeys[k], d.lever[k]);
  return o;
}

PopulationSpec parsePopulation(const val& p) {
  const std::string preset = str(p, "preset", "secondary");
  PopulationSpec s = preset == "children" ? PopulationSpec::children() : preset == "adults" ? PopulationSpec::adults() : PopulationSpec::secondary();
  s.age = num(p, "age", s.age);
  s.priorMean = num(p, "priorMean", s.priorMean);
  s.priorSd = num(p, "priorSd", s.priorSd);
  s.anxietyMean = num(p, "anxietyMean", s.anxietyMean);
  s.growthMean = num(p, "growthMean", s.growthMean);
  s.efficacyMean = num(p, "efficacyMean", s.efficacyMean);
  s.intrinsicMean = num(p, "intrinsicMean", s.intrinsicMean);
  s.capitalMean = num(p, "capitalMean", s.capitalMean);
  s.capitalSd = num(p, "capitalSd", s.capitalSd);
  return s;
}

CourseSpec parseCourse(const val& c) {
  CourseSpec s;
  s.sessions = integer(c, "sessions", s.sessions);
  s.sessionGap = num(c, "sessionGap", s.sessionGap);
  s.difficulty = num(c, "difficulty", s.difficulty);
  s.difficultyStep = num(c, "difficultyStep", s.difficultyStep);
  s.delayedTest = num(c, "delayedTest", s.delayedTest);
  s.abstraction = num(c, "abstraction", s.abstraction);
  s.atRiskBelow = num(c, "atRiskBelow", s.atRiskBelow);
  return s;
}

ClassroomEngine parseEngine(const val& spec, int maxLearners = 1000000) {
  const val sim = sub(spec, "sim");
  SimulationSettings s;
  s.learners = std::min(maxLearners, integer(sim, "learners", s.learners));
  s.seed = seedOf(spec);
  return ClassroomEngine(parsePopulation(sub(spec, "population")), parseCourse(sub(spec, "course")), s);
}

val profileToJs(const LearningProfile& p) {
  val o = val::object();
  o.set("days", arr(p.days));
  o.set("mean", arr(p.mean));
  o.set("q10", arr(p.q10));
  o.set("q50", arr(p.q50));
  o.set("q90", arr(p.q90));
  o.set("atRisk", arr(p.atRisk));
  return o;
}

val summaryToJs(const ClassroomResult& r) {
  val o = val::object();
  o.set("endScore", r.endScore);
  o.set("delayedScore", r.delayedScore);
  o.set("retention", r.retention);
  o.set("equityGap", r.equityGap);
  o.set("atRiskDelayed", r.atRiskDelayed);
  o.set("abilityGain", r.abilityGain);
  o.set("lastSessionDay", r.lastSessionDay);
  o.set("delayedTestDay", r.delayedTestDay);
  o.set("learners", r.learnerCount);
  return o;
}

// ------------------------------------------------------------------ entry points

val version(val) {
  val o = val::object();
  o.set("library", std::string("synaptic ") + SYN_VERSION);
  o.set("levers", kLevers);
  return o;
}

val coreDemo(val spec) {
  return guarded([&] {
    const double t0 = nowMs();
    const val c = sub(spec, "core");
    const int n = integer(c, "samples", 20000);
    Rng rng(seedOf(spec));
    std::vector<double> normals(n), uniforms(n);
    for (int i = 0; i < n; ++i) {
      normals[i] = rng.normal();
      uniforms[i] = rng.uniform();
    }
    const Summary s = summarise(normals);
    val o = val::object();
    o.set("normals", arr(normals));
    o.set("uniforms", arr(uniforms));
    val sm = val::object();
    sm.set("mean", s.mean);
    sm.set("sd", s.sd);
    sm.set("q10", s.q10);
    sm.set("median", s.median);
    sm.set("q90", s.q90);
    o.set("summary", sm);
    std::vector<double> xs = linspace(-6, 6, 121), ys;
    for (double x : xs) ys.push_back(logistic(x));
    o.set("logisticX", arr(xs));
    o.set("logisticY", arr(ys));
    // Common random numbers: two streams of the same seed agree, other seeds do not.
    Rng a = Rng::stream(seedOf(spec), 3), b = Rng::stream(seedOf(spec), 3), other = Rng::stream(seedOf(spec) + 1, 3);
    bool same = true, differ = true;
    for (int i = 0; i < 1000; ++i) {
      const auto x = a.next();
      same = same && x == b.next();
      differ = differ && x != other.next();
    }
    o.set("streamsReproducible", same);
    // Pure elliptic CGA: the axioms the models rely on, checked on random configurations.
    {
      const EllipticCGA g(3);
      Rng r(seedOf(spec) + 7);
      auto rnd = [&] { return std::vector<double>{r.normal(), r.normal(), r.normal(), r.normal()}; };
      double nullErr = 0, angleErr = 0, capErr = 0, rotorErr = 0, dilationErr = 0;
      for (int k = 0; k < 200; ++k) {
        const auto x = g.point(rnd()), y = g.point(rnd()), z = g.point(rnd());
        nullErr = std::max(nullErr, std::fabs(g.dot(x, x)));
        const double th = g.angle(x, y);
        angleErr = std::max(angleErr, std::fabs(g.dot(x, y) - (std::cos(th) - 1.0)));
        const double rho = 0.2 + 2.5 * r.uniform();
        capErr = std::max(capErr, std::fabs(g.capSide(y, g.cap(x, rho)) - (std::cos(th) - std::cos(rho))));
        const auto rot = g.rotorTowards(x, y, r.uniform());
        rotorErr = std::max(rotorErr, std::fabs(g.angle(g.move(rot, y), g.move(rot, z)) - g.angle(y, z)));
        const double beta = r.normal(), phi = g.angle(x, z);
        const auto moved = g.move(g.dilation(g.direction(x), beta), z);
        dilationErr = std::max(dilationErr, std::fabs(std::tan(g.angle(x, moved) / 2) - std::exp(beta) * std::tan(phi / 2)) /
                                                 std::max(1.0, std::exp(beta) * std::tan(phi / 2)));
      }
      val cga = val::object();
      cga.set("nullPoints", nullErr);
      cga.set("innerProductIsAngle", angleErr);
      cga.set("capMembership", capErr);
      cga.set("rotorIsometry", rotorErr);
      cga.set("dilationTangent", dilationErr);
      o.set("cga", cga);
    }
    o.set("seedsIndependent", differ);
    const auto uniform = SessionGrid::uniform(0, num(c, "gap", 7), integer(c, "sessions", 8));
    const auto expanding = SessionGrid::expanding(0, num(c, "firstGap", 1), num(c, "ratio", 2), integer(c, "sessions", 8));
    o.set("uniformGrid", arr(uniform.days()));
    o.set("expandingGrid", arr(expanding.days()));
    o.set("elapsedMs", nowMs() - t0);
    return o;
  });
}

val conditioning(val spec) {
  return guarded([&] {
    const val c = sub(spec, "conditioning");
    const double alpha = num(c, "salience", 0.3);
    const int acq = integer(c, "acquisition", 30), ext = integer(c, "extinction", 30);
    const auto pav = pavlovExperiment(alpha, acq, ext);
    val o = val::object();
    std::vector<double> bell, closed;
    for (std::size_t t = 0; t < pav.strengths.size(); ++t) {
      bell.push_back(pav.strengths[t][0]);
      closed.push_back(t < static_cast<std::size_t>(acq) ? acquisitionCurve(alpha, 0.5, 1.0, static_cast<int>(t) + 1) : NAN);
    }
    o.set("pavlov", arr(bell));
    o.set("closedForm", arr(closed));
    o.set("acquisitionTrials", acq);
    const auto bl = blockingExperiment(integer(c, "pretraining", 20), integer(c, "compound", 20));
    val b = val::object();
    std::vector<double> light, tone, ctrlTone;
    for (const auto& s : bl.blocked.strengths) {
      light.push_back(s[0]);
      tone.push_back(s[1]);
    }
    for (const auto& s : bl.control.strengths) ctrlTone.push_back(s[1]);
    b.set("light", arr(light));
    b.set("tone", arr(tone));
    b.set("controlTone", arr(ctrlTone));
    b.set("blockedTone", bl.blockedTone);
    b.set("controlToneFinal", bl.controlTone);
    b.set("compoundStart", bl.blocked.phaseStart[1]);
    o.set("blocking", b);
    const auto fear = fearConditioningExperiment(integer(c, "pairings", 7), num(c, "width", 1.0));
    val probes = val::array();
    for (const auto& p : fear.probes) {
      val q = val::object();
      q.set("stimulus", p.stimulus);
      q.set("distance", p.distance);
      q.set("fear", p.fear);
      probes.call<void>("push", q);
    }
    o.set("fear", probes);
    return o;
  });
}

val operant(val spec) {
  return guarded([&] {
    const val c = sub(spec, "operant");
    OperantParams p;
    p.acquisitionRate = num(c, "acquisitionRate", p.acquisitionRate);
    p.extinctionRate = num(c, "extinctionRate", p.extinctionRate);
    const int training = integer(c, "training", 300), extinction = integer(c, "extinction", 300);
    const int learners = integer(c, "learners", 200);
    const double n = num(c, "ratio", 5);
    val runs = val::array();
    for (const auto& s : {ReinforcementSchedule::continuous(), ReinforcementSchedule::fixedRatio(n), ReinforcementSchedule::variableRatio(n),
                          ReinforcementSchedule::fixedInterval(n), ReinforcementSchedule::variableInterval(n)}) {
      const auto r = simulateOperant(p, s, training, extinction, learners, seedOf(spec));
      val x = val::object();
      x.set("schedule", r.schedule);
      x.set("probability", arr(r.probability));
      x.set("cumulativeResponses", arr(r.cumulativeResponses));
      x.set("cumulativeReinforcers", arr(r.cumulativeReinforcers));
      x.set("acquisitionSteps", r.acquisitionSteps);
      x.set("resistance", r.resistance);
      x.set("reinforcersPerResponse", r.reinforcersPerResponse);
      runs.call<void>("push", x);
    }
    val o = val::object();
    o.set("training", training);
    o.set("runs", runs);
    return o;
  });
}

val tokenEconomy(val spec) {
  return guarded([&] {
    const val c = sub(spec, "tokens");
    TokenEconomyParams p;
    p.incentive = num(c, "incentive", p.incentive);
    p.undermining = num(c, "undermining", p.undermining);
    p.levelBonus = num(c, "levelBonus", p.levelBonus);
    p.students = integer(c, "students", 200);
    const auto r = simulateTokenEconomy(p, seedOf(spec));
    val o = val::object();
    o.set("engagement", arr(r.engagement));
    o.set("low", arr(r.engagementLow));
    o.set("high", arr(r.engagementHigh));
    o.set("points", arr(r.points));
    o.set("intrinsic", arr(r.intrinsic));
    o.set("before", r.before);
    o.set("during", r.during);
    o.set("after", r.after);
    o.set("rewardStart", p.rewardStart);
    o.set("rewardEnd", p.rewardEnd);
    return o;
  });
}

val humanism(val spec) {
  return guarded([&] {
    const val c = sub(spec, "humanism");
    // Maslow: readiness as safety and belonging rise, without and with a supportive climate.
    const auto grid = linspace(0, 1, 41);
    std::vector<double> plain, climate;
    for (double s : grid) {
      NeedState n;
      n.satisfaction = {0.9, s, s, 0.45, 0.2};
      plain.push_back(learningReadiness(n));
      climate.push_back(learningReadiness(applyClassroomClimate(n, 0.6, 0.6, 0.4)));
    }
    val o = val::object();
    o.set("needGrid", arr(grid));
    o.set("readiness", arr(plain));
    o.set("readinessWithClimate", arr(climate));
    NeedState student;
    student.satisfaction = {num(c, "physiological", 0.9), num(c, "safety", 0.5), num(c, "belonging", 0.45), num(c, "esteem", 0.4), 0.2};
    const NeedState supported = applyClassroomClimate(student, num(c, "empathy", 0.7), num(c, "belongingSupport", 0.6), num(c, "recognition", 0.5));
    o.set("student", arr(student.satisfaction));
    o.set("supported", arr(supported.satisfaction));
    o.set("activation", arr(needActivation(student)));
    o.set("activationSupported", arr(needActivation(supported)));
    o.set("dominant", std::string(needName(dominantNeed(student))));
    o.set("dominantSupported", std::string(needName(dominantNeed(supported))));
    val names = val::array();
    for (int k = 0; k < kNeedLevels; ++k) names.call<void>("push", std::string(needName(static_cast<Need>(k))));
    o.set("needs", names);
    // Kuhlthau's ISP at three levels of instructor support.
    val isp = val::array();
    for (double support : {0.0, 0.5, 1.0}) {
      IspParams p;
      p.support = support;
      p.anxietyMean = num(c, "anxiety", p.anxietyMean);
      p.deadline = integer(c, "deadline", p.deadline);
      p.students = integer(c, "students", p.students);
      const auto r = simulateInformationSearch(p, seedOf(spec));
      val x = val::object();
      x.set("support", support);
      x.set("days", arr(r.days));
      x.set("uncertainty", arr(r.meanUncertainty));
      val shares = val::array();
      for (const auto& s : r.stageShare) shares.call<void>("push", arr(s));
      x.set("stageShare", shares);
      x.set("daysInStage", arr(r.meanDaysInStage));
      x.set("completedOnTime", r.completedOnTime);
      x.set("satisfied", r.satisfied);
      x.set("meanCompletionDay", r.meanCompletionDay);
      isp.call<void>("push", x);
    }
    o.set("isp", isp);
    val stages = val::array();
    for (const auto& s : ispStages()) {
      val x = val::object();
      x.set("name", std::string(s.name));
      x.set("feeling", std::string(s.feeling));
      x.set("uncertainty", s.uncertainty);
      stages.call<void>("push", x);
    }
    o.set("stages", stages);
    o.set("deadline", integer(c, "deadline", 42));
    return o;
  });
}

val memory(val spec) {
  return guarded([&] {
    const val c = sub(spec, "memory");
    MemoryModel model;
    const int reviews = integer(c, "reviews", 4);
    const double gap = num(c, "gap", 3), testDay = num(c, "testDay", 30), horizon = num(c, "horizon", 45);
    const double cramDay = num(c, "cramDay", testDay - 1);
    std::vector<StudyPlan> plans = {
        massedPlan(StudyMode::Restudy, reviews, 0), massedPlan(StudyMode::Retrieval, reviews, 0), spacedPlan(StudyMode::Restudy, reviews, gap),
        spacedPlan(StudyMode::RetrievalWithFeedback, reviews, gap), expandingPlan(StudyMode::RetrievalWithFeedback, reviews, 1, 2),
        massedPlan(StudyMode::Restudy, reviews, cramDay)};
    plans.back().name = "cram: massed restudy the day before the test";
    val out = val::array();
    for (const auto& p : plans) {
      const auto r = simulatePlan(model, p, testDay, horizon, integer(c, "learners", 400), num(c, "heterogeneity", 0.3), seedOf(spec));
      val x = val::object();
      x.set("name", r.name);
      x.set("days", arr(r.days));
      x.set("recall", arr(r.meanRecall));
      x.set("q10", arr(r.q10Recall));
      x.set("q90", arr(r.q90Recall));
      x.set("recallAtTest", r.recallAtTest);
      x.set("stability", r.meanStabilityAtTest);
      x.set("successRate", r.reviewSuccessRate);
      std::vector<double> events;
      for (const auto& e : p.reviews) events.push_back(e.day);
      x.set("events", arr(events));
      out.call<void>("push", x);
    }
    val o = val::object();
    o.set("plans", out);
    o.set("testDay", testDay);
    // Stability gain of one study event against the retrievability at the time.
    const auto rs = linspace(0.05, 1.0, 40);
    std::vector<double> gr, gq;
    for (double r : rs) {
      gr.push_back(model.gain(StudyMode::Restudy, r));
      gq.push_back(model.gain(StudyMode::Retrieval, r));
    }
    o.set("gainR", arr(rs));
    o.set("gainRestudy", arr(gr));
    o.set("gainRetrieval", arr(gq));
    // Interleaving.
    InterleavingParams ip;
    ip.categories = integer(c, "categories", ip.categories);
    ip.problemsPerCategory = integer(c, "problemsPerCategory", ip.problemsPerCategory);
    const auto il = simulateInterleaving(ip, seedOf(spec));
    val inter = val::object();
    for (const auto& [name, r] : {std::pair<const char*, const PracticeOutcome*>{"blocked", &il.blocked}, {"interleaved", &il.interleaved}}) {
      val x = val::object();
      x.set("practice", arr(r->practiceAccuracy));
      x.set("practiceMean", r->practiceMean);
      x.set("test", r->testAccuracy);
      x.set("choice", r->choiceAccuracy);
      x.set("execution", r->execution);
      inter.set(name, x);
    }
    o.set("interleaving", inter);
    // Schemas: AND encoded with and without stimulated recall of prior knowledge.
    auto plain = SchemaNetwork::librarySearchExample();
    auto primed = SchemaNetwork::librarySearchExample();
    const auto venn = plain.index("Venn diagram");
    const auto a1 = plain.learn("AND operator", {venn}, {});
    const auto a2 = primed.learn("AND operator", {venn}, {primed.index("narrow results"), primed.index("keyword"), primed.index("research question")});
    auto networkJs = [&](const SchemaNetwork& s, std::size_t target) {
      val x = val::object();
      std::vector<std::string> names;
      for (std::size_t i = 0; i < s.size(); ++i) names.push_back(s.name(i));
      x.set("nodes", strings(names));
      val edges = val::array();
      for (std::size_t i = 0; i < s.size(); ++i)
        for (std::size_t j = i + 1; j < s.size(); ++j)
          if (s.weight(i, j) > 0.0) {
            val e = val::array();
            e.call<void>("push", static_cast<double>(i));
            e.call<void>("push", static_cast<double>(j));
            e.call<void>("push", s.weight(i, j));
            edges.call<void>("push", e);
          }
      x.set("edges", edges);
      std::vector<double> recall;
      for (std::size_t i = 0; i < s.size(); ++i) recall.push_back(s.recallProbability(target, {i}));
      x.set("recallFrom", arr(recall));
      x.set("degree", static_cast<double>(s.degree(target)));
      return x;
    };
    o.set("schemaPlain", networkJs(plain, a1));
    o.set("schemaPrimed", networkJs(primed, a2));
    return o;
  });
}

val equilibration(val spec) {
  return guarded([&] {
    const val c = sub(spec, "equilibration");
    EquilibrationParams base;
    base.accommodationThreshold = num(c, "threshold", base.accommodationThreshold);
    AnimalWorld w;
    w.observations = integer(c, "observations", w.observations);
    w.learners = integer(c, "learners", w.learners);
    val runs = val::array();
    auto push = [&](const std::string& label, const EquilibrationParams& p, double feedback) {
      AnimalWorld ww = w;
      ww.feedback = feedback;
      const auto r = simulateEquilibration(p, ww, seedOf(spec));
      val x = val::object();
      x.set("label", label);
      x.set("feedback", feedback);
      x.set("avoidance", p.avoidance);
      x.set("accuracy", arr(r.accuracy));
      x.set("disequilibrium", arr(r.disequilibrium));
      x.set("schemas", arr(r.schemaCount));
      x.set("finalAccuracy", r.finalAccuracy);
      x.set("assimilations", r.assimilations);
      x.set("accommodations", r.accommodations);
      x.set("avoidances", r.avoidances);
      x.set("conflicts", r.conflicts);
      runs.call<void>("push", x);
    };
    const double feedback = num(c, "feedback", 0.3);
    push("no one names the animals", base, 0.0);
    push("adults sometimes name them", base, feedback);
    push("adults always name them", base, 1.0);
    EquilibrationParams avoid = base;
    avoid.avoidance = num(c, "avoidance", 0.7);
    push("named, but discomfort is avoided", avoid, feedback);
    val o = val::object();
    o.set("runs", runs);
    o.set("catsFrom", w.catsFrom);
    return o;
  });
}

val zpd(val spec) {
  return guarded([&] {
    const val c = sub(spec, "zpd");
    ZpdParams z;
    z.scaffold = num(c, "scaffold", z.scaffold);
    const double theta = num(c, "theta", 0.0);
    const auto bs = linspace(theta - 5, theta + 7, 121);
    std::vector<double> alone, assisted, gain, zone;
    for (double b : bs) {
      alone.push_back(successAlone(theta, b));
      assisted.push_back(successAssisted(theta, b, z.scaffold));
      gain.push_back(expectedGain(theta, b, z));
      zone.push_back(static_cast<double>(zoneOf(theta, b, z)));
    }
    val o = val::object();
    o.set("difficulty", arr(bs));
    o.set("alone", arr(alone));
    o.set("assisted", arr(assisted));
    o.set("gain", arr(gain));
    o.set("zone", arr(zone));
    const auto band = zpdBand(theta, z);
    o.set("bandLow", band.first);
    o.set("bandHigh", band.second);
    // Figure 3.2: tasks on the sphere around the learner, zones from the CGA caps, drawn by
    // stereographic projection (concentric circles).
    std::vector<std::vector<double>> mx(3), my(3);
    for (const auto& m : zpdMap(z, integer(c, "rings", 22), integer(c, "perRing", 24), -4.0, 3.5)) {
      mx[static_cast<int>(m.zone)].push_back(m.x);
      my[static_cast<int>(m.zone)].push_back(m.y);
    }
    o.set("mapX", arrOfArr(mx));
    o.set("mapY", arrOfArr(my));
    const auto radii = zpdMapRadii(z);
    o.set("mapInner", radii.first);
    o.set("mapOuter", radii.second);
    TutoringParams tp;
    tp.sessions = integer(c, "sessions", tp.sessions);
    tp.learners = integer(c, "learners", tp.learners);
    tp.fadeScaffold = flag(c, "fade", true);
    val runs = val::array();
    for (auto policy : {TaskPolicy::TooEasy, TaskPolicy::ClassAverage, TaskPolicy::TooHard, TaskPolicy::Zpd}) {
      const auto r = simulateTutoring(z, tp, policy, seedOf(spec));
      val x = val::object();
      x.set("policy", r.policy);
      x.set("mean", arr(r.meanAbility));
      x.set("q10", arr(r.q10Ability));
      x.set("q90", arr(r.q90Ability));
      x.set("zoneShare", arrOfArr(r.zoneShare));
      x.set("gain", r.finalGain);
      runs.call<void>("push", x);
    }
    o.set("tutoring", runs);
    return o;
  });
}

val dialogue(val spec) {
  return guarded([&] {
    const val c = sub(spec, "dialogue");
    val runs = val::array();
    for (auto g : {Grouping::None, Grouping::Random, Grouping::MixedAbility, Grouping::SameAbility})
      for (double inclusivity : {0.0, num(c, "inclusivity", 1.0)}) {
        DialogueParams p;
        p.grouping = g;
        p.inclusivity = inclusivity;
        p.groupSize = integer(c, "groupSize", p.groupSize);
        p.sessions = integer(c, "sessions", p.sessions);
        p.capitalPenalty = num(c, "capitalPenalty", p.capitalPenalty);
        const auto r = simulateDialogue(p, seedOf(spec));
        val x = val::object();
        x.set("grouping", r.grouping);
        x.set("inclusivity", inclusivity);
        x.set("mean", arr(r.meanUnderstanding));
        x.set("high", arr(r.highCapital));
        x.set("low", arr(r.lowCapital));
        x.set("spread", arr(r.spread));
        x.set("finalMean", r.finalMean);
        x.set("finalGap", r.finalGap);
        runs.call<void>("push", x);
      }
    val o = val::object();
    o.set("runs", runs);
    return o;
  });
}

val development(val spec) {
  return guarded([&] {
    const val c = sub(spec, "development");
    const double sd = num(c, "onsetSd", 0.9);
    const auto ages = linspace(0, 18, 181);
    std::vector<std::vector<double>> dist(kPiagetStages);
    for (double a : ages) {
      const auto p = stageDistribution(a, sd);
      for (int k = 0; k < kPiagetStages; ++k) dist[k].push_back(p[k]);
    }
    val o = val::object();
    o.set("ages", arr(ages));
    o.set("stageDistribution", arrOfArr(dist));
    val stages = val::array();
    for (const auto& s : piagetStages()) {
      val x = val::object();
      x.set("name", std::string(s.name));
      x.set("ages", std::string(s.ages));
      stages.call<void>("push", x);
    }
    o.set("stages", stages);
    val acts = val::array();
    for (const auto& a : climateChangeLesson()) {
      std::vector<double> fit;
      for (double age : ages) fit.push_back(activityFit(a, age, sd));
      val x = val::object();
      x.set("name", a.name);
      x.set("fit", arr(fit));
      std::vector<std::string> demands;
      for (auto d : a.demands) demands.push_back(capabilityName(d));
      x.set("demands", strings(demands));
      acts.call<void>("push", x);
    }
    o.set("activities", acts);
    PerryParams pp;
    const int terms = integer(c, "terms", 8);
    auto progression = [&](double challenge) {
      PerryParams q = pp;
      q.challenge = challenge;
      const auto path = perryProgression({1, 0, 0, 0}, q, terms);
      std::vector<std::vector<double>> byPos(kPerryPositions);
      for (const auto& d : path)
        for (int k = 0; k < kPerryPositions; ++k) byPos[k].push_back(d[k]);
      return arrOfArr(byPos);
    };
    o.set("perryPlain", progression(0.0));
    o.set("perryChallenged", progression(num(c, "challenge", 1.0)));
    std::vector<double> acc;
    std::vector<std::string> names;
    for (int k = 0; k < kPerryPositions; ++k) {
      names.push_back(perryName(static_cast<PerryPosition>(k)));
      acc.push_back(sourceEvaluationAccuracy(static_cast<PerryPosition>(k), 8, 200, num(c, "authorityValidity", 0.3), seedOf(spec)));
    }
    o.set("perryNames", strings(names));
    o.set("sourceAccuracy", arr(acc));
    return o;
  });
}

val andragogy(val spec) {
  return guarded([&] {
    const val c = sub(spec, "andragogy");
    val o = val::object();
    const auto mix = linspace(0, 1, 21);
    const LearnerOrientation learners[3] = {LearnerOrientation::child(), LearnerOrientation::adolescent(), LearnerOrientation::adult()};
    const auto ped = AndragogicalDesign::pedagogical(), andr = AndragogicalDesign::andragogical();
    std::vector<std::vector<double>> curves;
    for (const auto& l : learners) {
      std::vector<double> e;
      for (double m : mix) {
        AndragogicalDesign d;
        for (int k = 0; k < kAndragogyDimensions; ++k) d.feature[k] = (1 - m) * ped.feature[k] + m * andr.feature[k];
        e.push_back(engagement(l, d));
      }
      curves.push_back(e);
    }
    o.set("mix", arr(mix));
    o.set("curves", arrOfArr(curves));
    std::vector<std::vector<double>> terms;
    for (const auto& l : learners) {
      const auto tp = engagementTerms(l, ped), ta = engagementTerms(l, andr);
      std::vector<double> diff;
      for (int k = 0; k < kAndragogyDimensions; ++k) diff.push_back(ta[k] - tp[k]);
      terms.push_back(diff);
    }
    o.set("termShift", arrOfArr(terms));
    std::vector<std::string> assumptions, features, steps;
    for (int k = 0; k < kAndragogyDimensions; ++k) {
      assumptions.push_back(assumptionName(k));
      features.push_back(designFeatureName(k));
    }
    for (int k = 0; k < kAndragogySteps; ++k) steps.push_back(andragogyStepName(static_cast<AndragogyStep>(k)));
    o.set("assumptions", strings(assumptions));
    o.set("features", strings(features));
    o.set("steps", strings(steps));
    val cycles = val::array();
    for (double input : {0.0, 0.5, 1.0}) {
      NeedsCycleParams p;
      p.learnerInput = input;
      p.climate = input;
      p.cycles = integer(c, "cycles", p.cycles);
      const auto r = simulateNeedsCycle(p, seedOf(spec));
      val x = val::object();
      x.set("input", input);
      x.set("gap", arr(r.meanGap));
      x.set("hitRate", arr(r.hitRate));
      cycles.call<void>("push", x);
    }
    o.set("cycles", cycles);
    return o;
  });
}

val motivation(val spec) {
  return guarded([&] {
    const val c = sub(spec, "motivation");
    ExpectancyValueParams ep;
    ep.weeks = integer(c, "weeks", ep.weeks);
    val o = val::object();
    val runs = val::array();
    auto term = [&](const std::string& label, const MotivationLevers& lv) {
      const auto r = simulateTerm(lv, ep, seedOf(spec));
      val x = val::object();
      x.set("label", label);
      x.set("motivation", arr(r.motivation));
      x.set("value", arr(r.value));
      x.set("efficacy", arr(r.efficacy));
      x.set("ability", arr(r.ability));
      x.set("calibration", arr(r.calibrationError));
      x.set("engaged", arr(r.engaged));
      x.set("gain", r.finalAbilityGain);
      x.set("finalMotivation", r.finalMotivation);
      runs.call<void>("push", x);
    };
    term("no levers", {});
    for (int k = 0; k < MotivationLevers::kCount; ++k) {
      MotivationLevers one;
      one[k] = 1.0;
      term(MotivationLevers::name(k), one);
    }
    MotivationLevers all;
    for (int k = 0; k < MotivationLevers::kCount; ++k) all[k] = 1.0;
    term("all five", all);
    o.set("terms", runs);
    std::vector<double> off = linspace(-3, 4, 71), interest;
    for (double x : off) interest.push_back(challengeInterest(x));
    o.set("offset", arr(off));
    o.set("interest", arr(interest));
    val minds = val::array();
    for (auto f : {FeedbackStyle::Ability, FeedbackStyle::Neutral, FeedbackStyle::ProcessNotYet}) {
      MindsetParams mp;
      mp.feedback = f;
      mp.weeks = integer(c, "mindsetWeeks", mp.weeks);
      mp.coaching = num(c, "coaching", 0.0);
      mp.highStandards = flag(c, "highStandards", true);
      const auto r = simulateMindset(mp, seedOf(spec));
      val x = val::object();
      x.set("feedback", std::string(feedbackStyleName(f)));
      x.set("skillFixed", arr(r.skillFixed));
      x.set("skillGrowth", arr(r.skillGrowth));
      x.set("persisting", arr(r.persisting));
      x.set("mindset", arr(r.mindset));
      x.set("challenge", arr(r.challengeSeeking));
      x.set("gain", r.finalSkillGain);
      x.set("persist", r.finalPersisting);
      minds.call<void>("push", x);
    }
    o.set("mindset", minds);
    return o;
  });
}

// ------------------------------------------------------------------ the classroom analyses
//
// classroom, theories, allocation and sensitivity run on any of three engines: the
// ClassroomEngine here (WebAssembly), the fused WebGPU kernels (gpuJobs → GPU → gpuAnalyse) or
// the same kernels on the CPU (cpuKernelSlices on every worker → cpuKernelAnalyse). The result
// objects are built by the same functions whichever engine produced the numbers.

val classroomJs(const ClassroomResult& r) {
  val o = summaryToJs(r);
  o.set("score", profileToJs(r.score));
  o.set("recall", profileToJs(r.recall));
  val s = val::object();
  const auto& t = r.sessions;
  s.set("day", arr(t.day));
  s.set("ability", arr(t.ability));
  s.set("effort", arr(t.effort));
  s.set("motivation", arr(t.motivation));
  s.set("value", arr(t.value));
  s.set("efficacy", arr(t.efficacy));
  s.set("anxiety", arr(t.anxiety));
  s.set("readiness", arr(t.readiness));
  s.set("growth", arr(t.growth));
  s.set("engaged", arr(t.engaged));
  s.set("zpdShare", arr(t.zpdShare));
  o.set("sessions", s);
  return o;
}

val theoriesJs(const std::vector<std::string>& populations, const std::vector<std::vector<ClassroomResult>>& results) {
  val o = val::object();
  std::vector<std::string> names;
  val designs = val::array();
  for (int t = 0; t < kTheories; ++t) {
    names.push_back(theoryName(static_cast<Theory>(t)));
    designs.call<void>("push", designToJs(theoryDesign(static_cast<Theory>(t))));
  }
  o.set("theories", strings(names));
  o.set("designs", designs);
  val pops = val::array();
  for (std::size_t i = 0; i < populations.size(); ++i) {
    val p = val::object();
    p.set("population", populations[i]);
    val list = val::array();
    for (const auto& r : results[i]) {
      val x = summaryToJs(r);
      x.set("days", arr(r.score.days));
      x.set("mean", arr(r.score.mean));
      list.call<void>("push", x);
    }
    p.set("results", list);
    pops.call<void>("push", p);
  }
  o.set("populations", pops);
  return o;
}

val allocationJs(const AllocationResult& r) {
  val o = val::object();
  o.set("baselineScore", r.baselineScore);
  o.set("designScore", r.designScore);
  o.set("totalGain", r.totalGain);
  o.set("shapleySum", r.shapleySum);
  o.set("permutations", r.permutations);
  o.set("evaluations", r.evaluations);
  val levers = val::array();
  for (const auto& c : r.levers) {
    val x = val::object();
    x.set("lever", std::string(leverName(c.lever)));
    x.set("theory", std::string(leverTheory(c.lever)));
    x.set("baseline", c.baselineValue);
    x.set("design", c.designValue);
    x.set("shapley", c.shapley);
    x.set("incremental", c.incremental);
    x.set("standalone", c.standalone);
    levers.call<void>("push", x);
  }
  o.set("levers", levers);
  return o;
}

val sensitivityJs(const SensitivityResult& r, int learners) {
  val o = val::object();
  o.set("score", r.score);
  o.set("bump", r.bump);
  o.set("replications", r.replications);
  o.set("learners", learners);
  o.set("varianceReduction", r.meanVarianceReduction);
  val levers = val::array();
  for (const auto& l : r.levers) {
    val x = val::object();
    x.set("lever", std::string(leverName(l.lever)));
    x.set("theory", std::string(leverTheory(l.lever)));
    x.set("value", l.value);
    x.set("crn", l.crn);
    x.set("crnSe", l.crnStdError);
    x.set("independent", l.independent);
    x.set("independentSe", l.independentStdError);
    levers.call<void>("push", x);
  }
  o.set("levers", levers);
  return o;
}

const std::vector<PopulationSpec>& theoryPopulations() {
  static const std::vector<PopulationSpec> p = {PopulationSpec::children(), PopulationSpec::secondary(), PopulationSpec::adults()};
  return p;
}

AllocationRequest allocationRequestOf(const val& spec) {
  const val a = sub(spec, "allocation");
  const InstructionalDesign baseline = has(a, "baseline") ? parseDesign(a["baseline"]) : theoryDesign(Theory::Banking);
  return allocationRequest(parseDesign(sub(spec, "design")), baseline, integer(a, "permutations", 12));
}

int sensitivityLearners(const val& spec) {
  return std::min(integer(sub(spec, "sensitivity"), "maxLearners", 300), integer(sub(spec, "sim"), "learners", 500));
}

SensitivityRequest sensitivityRequestOf(const val& spec) {
  const val s = sub(spec, "sensitivity");
  return sensitivityRequest(parseDesign(sub(spec, "design")), num(s, "bump", 0.1), integer(s, "replications", 6), seedOf(spec));
}

// ---- WebAssembly engine

val classroom(val spec) {
  return guarded([&] {
    const double t0 = nowMs();
    const auto engine = parseEngine(spec);
    const auto r = engine.run(parseDesign(sub(spec, "design")));
    val o = classroomJs(r);
    // Per-learner views: only this engine keeps individual learners.
    const std::size_t sample = std::min<std::size_t>(r.learners.size(), static_cast<std::size_t>(integer(spec, "scatterLearners", 400)));
    std::vector<double> capital, delayed, gain, all;
    for (std::size_t i = 0; i < sample; ++i) {
      capital.push_back(r.learners[i].capital);
      delayed.push_back(r.learners[i].delayedScore);
      gain.push_back(r.learners[i].abilityGain);
    }
    for (const auto& l : r.learners) all.push_back(l.delayedScore);
    o.set("capital", arr(capital));
    o.set("delayed", arr(delayed));
    o.set("gain", arr(gain));
    o.set("delayedAll", arr(all));
    o.set("elapsedMs", nowMs() - t0);
    return o;
  });
}

val theories(val spec) {
  return guarded([&] {
    const double t0 = nowMs();
    const val sim = sub(spec, "sim");
    SimulationSettings s;
    s.learners = integer(sim, "learners", 500);
    s.seed = seedOf(spec);
    const CourseSpec course = parseCourse(sub(spec, "course"));
    std::vector<std::string> names;
    std::vector<std::vector<ClassroomResult>> results;
    for (const auto& pop : theoryPopulations()) {
      ClassroomEngine engine(pop, course, s);
      names.push_back(pop.name);
      results.emplace_back();
      for (int t = 0; t < kTheories; ++t) results.back().push_back(engine.run(theoryDesign(static_cast<Theory>(t))));
    }
    val o = theoriesJs(names, results);
    o.set("elapsedMs", nowMs() - t0);
    return o;
  });
}

val gagne(val spec) {
  return guarded([&] {
    const auto engine = parseEngine(spec);
    std::array<double, kGagneEvents> all{};
    all.fill(1.0);
    const double full = engine.objective(designFromGagne(all));
    val events = val::array();
    for (int k = 0; k < kGagneEvents; ++k) {
      auto missing = all;
      missing[k] = 0.0;
      val x = val::object();
      x.set("event", std::string(gagneEventName(static_cast<GagneEvent>(k))));
      x.set("withoutIt", engine.objective(designFromGagne(missing)));
      events.call<void>("push", x);
    }
    std::array<double, kGagneEvents> lecture{};
    lecture[static_cast<int>(GagneEvent::PresentStimulus)] = 1.0;
    val o = val::object();
    o.set("full", full);
    o.set("presentOnly", engine.objective(designFromGagne(lecture)));
    o.set("events", events);
    o.set("design", designToJs(designFromGagne(all)));
    return o;
  });
}

val allocation(val spec) {
  return guarded([&] {
    const double t0 = nowMs();
    const auto engine = parseEngine(spec);
    const auto q = allocationRequestOf(spec);
    std::vector<double> scores;
    for (const auto& d : q.designs) scores.push_back(engine.objective(d));
    val o = allocationJs(allocateFromScores(q, scores));
    o.set("elapsedMs", nowMs() - t0);
    return o;
  });
}

val sensitivity(val spec) {
  return guarded([&] {
    const double t0 = nowMs();
    const auto engine = parseEngine(spec, sensitivityLearners(spec));
    const auto q = sensitivityRequestOf(spec);
    std::vector<double> scores;
    for (const auto& run : q.runs) scores.push_back(engine.objective(run.design, run.seed));
    val o = sensitivityJs(sensitivitiesFromScores(q, scores), engine.settings().learners);
    o.set("elapsedMs", nowMs() - t0);
    return o;
  });
}

// ---- fused kernels (WebGPU, or the same kernels on the CPU)

// The batches of an analysis, compiled for the fused kernels.
std::vector<gpu::FusedPlan> fusedPlans(const val& spec) {
  const std::string analysis = str(spec, "analysis", "classroom");
  const CourseSpec course = parseCourse(sub(spec, "course"));
  const int learners = integer(sub(spec, "sim"), "learners", 500);
  const auto seed = static_cast<std::uint32_t>(seedOf(spec));
  std::vector<gpu::FusedPlan> plans;
  if (analysis == "classroom") {
    plans.push_back(gpu::compile(parsePopulation(sub(spec, "population")), course, learners, {{parseDesign(sub(spec, "design")), seed}}, true, true));
  } else if (analysis == "theories") {
    std::vector<gpu::FusedJob> jobs;
    for (int t = 0; t < kTheories; ++t) jobs.push_back({theoryDesign(static_cast<Theory>(t)), seed});
    for (const auto& pop : theoryPopulations()) plans.push_back(gpu::compile(pop, course, learners, jobs, true, false));
  } else if (analysis == "allocation") {
    std::vector<gpu::FusedJob> jobs;
    for (const auto& d : allocationRequestOf(spec).designs) jobs.push_back({d, seed});
    plans.push_back(gpu::compile(parsePopulation(sub(spec, "population")), course, learners, jobs, false, false));
  } else if (analysis == "sensitivity") {
    std::vector<gpu::FusedJob> jobs;
    for (const auto& run : sensitivityRequestOf(spec).runs) jobs.push_back({run.design, static_cast<std::uint32_t>(run.seed)});
    plans.push_back(gpu::compile(parsePopulation(sub(spec, "population")), course, sensitivityLearners(spec), jobs, false, false));
  } else {
    throw std::invalid_argument("the fused kernels do not run the analysis " + analysis);
  }
  return plans;
}

// The analysis result from the kernels' read-backs, one FusedOutput per plan.
val fusedAnalyse(const val& spec, const std::vector<gpu::FusedPlan>& plans, const std::vector<gpu::FusedOutput>& outputs) {
  const std::string analysis = str(spec, "analysis", "classroom");
  if (outputs.size() != plans.size()) throw std::invalid_argument("one read-back per plan expected");
  if (analysis == "classroom") return classroomJs(gpu::summarise(plans[0], outputs[0], 0));
  if (analysis == "theories") {
    std::vector<std::string> names;
    std::vector<std::vector<ClassroomResult>> results;
    for (std::size_t i = 0; i < plans.size(); ++i) {
      names.push_back(theoryPopulations()[i].name);
      results.emplace_back();
      for (std::size_t j = 0; j < plans[i].numJobs; ++j) results.back().push_back(gpu::summarise(plans[i], outputs[i], j));
    }
    return theoriesJs(names, results);
  }
  if (analysis == "allocation") return allocationJs(allocateFromScores(allocationRequestOf(spec), gpu::delayedScores(plans[0], outputs[0])));
  return sensitivityJs(sensitivitiesFromScores(sensitivityRequestOf(spec), gpu::delayedScores(plans[0], outputs[0])),
                       static_cast<int>(plans[0].numLearners));
}

template <class T>
std::vector<T> typedVec(const val& v) {
  return emscripten::convertJSArrayToNumberVector<T>(v);
}

val gpuKernels(val) {
  val o = val::object();
  o.set("classroomFused", gpu::classroomFusedKernel());
  o.set("reducePartials", gpu::reducePartialsKernel());
  o.set("scoreQuantiles", gpu::scoreQuantilesKernel());
  return o;
}

val gpuJobs(val spec) {
  return guarded([&] {
    const double t0 = nowMs();
    const std::string why = gpu::limitation(parseCourse(sub(spec, "course")), static_cast<std::size_t>(integer(sub(spec, "sim"), "learners", 500)));
    if (!why.empty()) {
      val o = val::object();
      o.set("unsupported", why);
      return o;
    }
    const auto plans = fusedPlans(spec);
    val list = val::array();
    for (const auto& p : plans) {
      // The kernel reads one constant buffer: params | days | jobs.
      std::vector<float> constants = p.params;
      constants.insert(constants.end(), p.dayTable.begin(), p.dayTable.end());
      constants.insert(constants.end(), p.jobs.begin(), p.jobs.end());
      val x = val::object();
      x.set("header", typedArray(p.header));
      x.set("constants", typedArray(constants));
      x.set("seeds", typedArray(p.seeds));
      val dispatch = val::object();
      dispatch.set("fused", static_cast<double>(p.fusedDispatch()));
      dispatch.set("jobs", static_cast<double>(p.numJobs));
      dispatch.set("reduce", static_cast<double>(p.reduceDispatch()));
      dispatch.set("quantiles", static_cast<double>(p.quantileDispatch()));
      x.set("dispatch", dispatch);
      val bytes = val::object();
      bytes.set("scratch", static_cast<double>(p.scratchBytes()));
      bytes.set("partials", static_cast<double>(p.partialsBytes()));
      bytes.set("sums", static_cast<double>(p.sumsBytes()));
      bytes.set("quantiles", static_cast<double>(p.quantileBytes()));
      x.set("bytes", bytes);
      x.set("numJobs", static_cast<double>(p.numJobs));
      x.set("numLearners", static_cast<double>(p.numLearners));
      x.set("numDays", static_cast<double>(p.numDays()));
      list.call<void>("push", x);
    }
    val o = val::object();
    o.set("plans", list);
    o.set("compileMs", nowMs() - t0);
    return o;
  });
}

val gpuAnalyse(val spec) {
  return guarded([&] {
    const auto plans = fusedPlans(spec);
    std::vector<gpu::FusedOutput> outputs;
    const val reads = spec["gpuOutputs"];
    for (unsigned i = 0; i < reads["length"].as<unsigned>(); ++i) {
      gpu::FusedOutput out;
      out.sums = typedVec<float>(reads[i]["sums"]);
      if (has(reads[i], "quantiles")) out.quantiles = typedVec<float>(reads[i]["quantiles"]);
      if (!plans[i].quantiles) out.quantiles.clear();
      outputs.push_back(std::move(out));
    }
    return fusedAnalyse(spec, plans, outputs);
  });
}

val cpuKernelSlices(val spec) {
  return guarded([&] {
    const double t0 = nowMs();
    const auto plans = fusedPlans(spec);
    const int part = integer(spec, "part", 0), parts = std::max(1, integer(spec, "parts", 1));
    val list = val::array();
    for (const auto& p : plans) {
      const std::size_t tiles = p.numJobs * p.numWorkgroups();
      const std::size_t begin = tiles * part / parts, end = tiles * (part + 1) / parts;
      const auto slice = gpu::runFusedTiles(p, begin, end);
      val x = val::object();
      x.set("tileBegin", static_cast<double>(slice.tileBegin));
      x.set("tileEnd", static_cast<double>(slice.tileEnd));
      x.set("partials", typedArray(slice.partials));
      x.set("scores", typedArray(slice.scores));
      list.call<void>("push", x);
    }
    val o = val::object();
    o.set("slices", list);
    o.set("elapsedMs", nowMs() - t0);
    return o;
  });
}

val cpuKernelAnalyse(val spec) {
  return guarded([&] {
    const auto plans = fusedPlans(spec);
    const val parts = spec["parts"];
    std::vector<gpu::FusedOutput> outputs;
    for (std::size_t i = 0; i < plans.size(); ++i) {
      std::vector<gpu::FusedSlice> slices;
      for (unsigned k = 0; k < parts["length"].as<unsigned>(); ++k) {
        const val x = parts[k]["slices"][i];
        gpu::FusedSlice s;
        s.tileBegin = static_cast<std::size_t>(x["tileBegin"].as<double>());
        s.tileEnd = static_cast<std::size_t>(x["tileEnd"].as<double>());
        s.partials = typedVec<float>(x["partials"]);
        s.scores = typedVec<float>(x["scores"]);
        slices.push_back(std::move(s));
      }
      outputs.push_back(gpu::finishFused(plans[i], slices));
    }
    return fusedAnalyse(spec, plans, outputs);
  });
}

// Runs the kernels' CPU reference in this worker (Node tests of the GPU path).
val gpuEmulate(val spec) {
  return guarded([&] {
    const auto plans = fusedPlans(spec);
    std::vector<gpu::FusedOutput> outputs;
    for (const auto& p : plans) outputs.push_back(gpu::runFusedReference(p));
    return fusedAnalyse(spec, plans, outputs);
  });
}

val presets(val) {
  return guarded([&] {
    val o = val::object();
    for (int t = 0; t < kTheories; ++t) o.set(theoryName(static_cast<Theory>(t)), designToJs(theoryDesign(static_cast<Theory>(t))));
    return o;
  });
}

}  // namespace

EMSCRIPTEN_BINDINGS(syn) {
  emscripten::function("version", &version);
  emscripten::function("presets", &presets);
  emscripten::function("coreDemo", &coreDemo);
  emscripten::function("conditioning", &conditioning);
  emscripten::function("operant", &operant);
  emscripten::function("tokenEconomy", &tokenEconomy);
  emscripten::function("humanism", &humanism);
  emscripten::function("memory", &memory);
  emscripten::function("equilibration", &equilibration);
  emscripten::function("zpd", &zpd);
  emscripten::function("dialogue", &dialogue);
  emscripten::function("development", &development);
  emscripten::function("andragogy", &andragogy);
  emscripten::function("motivation", &motivation);
  emscripten::function("classroom", &classroom);
  emscripten::function("theories", &theories);
  emscripten::function("gagne", &gagne);
  emscripten::function("allocation", &allocation);
  emscripten::function("sensitivity", &sensitivity);
  emscripten::function("gpuKernels", &gpuKernels);
  emscripten::function("gpuJobs", &gpuJobs);
  emscripten::function("gpuAnalyse", &gpuAnalyse);
  emscripten::function("gpuEmulate", &gpuEmulate);
  emscripten::function("cpuKernelSlices", &cpuKernelSlices);
  emscripten::function("cpuKernelAnalyse", &cpuKernelAnalyse);
}
