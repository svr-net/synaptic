// Formative and summative assessment in a session, and the assessment cycle (chapter 9), in the
// space.
//
// A session teaches an outcome in chunks that build on each other. A learner's task in chunk j is
// the point of logit θ + c_j + κ(m_{j−1} − 1) − b_j (c_j how clearly the chunk was taught, unknown
// to the instructor), and their mastery the chance of the task dilated by the session's support,
// which grows with the minutes the chunk gets (with diminishing returns). A formative check after
// each chunk reads the class's mastery — informally (quick, flattering) or formally (slower,
// accurate) — and a chunk that looks unlearned is retaught, clearer, from a reteach kept in hand
// (unused, it becomes practice at the end). A summative assessment at the end tells the
// instructor whether the outcome was met.
// The assessment cycle (Figure 9.1): an agent is an instructor with one unclear part of the
// lesson, who teaches a few classes per offering, reads the evidence (direct, self-report,
// grades or none), revises the chunks it shows as gaps (clearer by a share of how weak they
// really were) and makes the outcome more challenging or more achievable.
#include <string>

#include "syn/models/models.hpp"
#include "syn/kernel/agent.hpp"

SYN_AGENT(session_agent,
  let isCycle = J(0u) > f32(0.5);
  let formative = select(J(1u), f32(0), isCycle);
  let summative = !isCycle && J(2u) > f32(0.5);
  let evidence = J(1u);
  let n = u32(P(0u));
  let chunks = u32(P(1u));
  let minutes = P(2u);
  let between = P(3u);
  let within = P(4u);
  let clarityMean = P(5u);
  let claritySd = P(6u);
  let scaffold = P(7u);
  let prerequisite = P(8u);
  let reteachThreshold = P(10u);
  let reteachMinutes = P(11u);
  let reteachClarity = P(12u);
  let threshold = P(13u);
  let offerings = select(1u, u32(P(20u)), isCycle);
  let classes = select(1u, u32(P(21u)), isCycle);
  let checking = formative > f32(0.5);
  let informal = formative > f32(0.5) && formative < f32(1.5);
  let checkMinutes = select(select(f32(0), f32(1), checking), f32(2), formative > f32(1.5));
  let assessMinutes = checkMinutes * f32(chunks) + select(f32(0), f32(5), summative);
  let nominal = minutes / f32(chunks);
  ARR(clarity, f32, 8);
  for (var j = 0u; j < chunks; j = j + 1u) { clarity[j] = P(24u + j); }
  var difficulty = P(9u);
  ARR(theta, f32, 32);
  ARR(prev, f32, 32);
  ARR(m, f32, 32);
  ARR(c, f32, 8);
  ARR(chunkMet, f32, 8);
  for (var o = 0u; o < offerings; o = o + 1u) {
    for (var j = 0u; j < chunks; j = j + 1u) { chunkMet[j] = f32(0); }
    var metSum = f32(0);
    var masterySum = f32(0);
    for (var cl = 0u; cl < classes; cl = cl + 1u) {
      let base = (o * classes + cl) * 512u;
      let mu = between * kLogisticUnit * logisticOf(U(base));
      for (var i = 0u; i < n; i = i + 1u) {
        theta[i] = logit(sampleLevel(U(base + 1u + i), mu, within, kA));
        prev[i] = f32(1);
      }
      for (var j = 0u; j < chunks; j = j + 1u) {
        c[j] = select(clarityMean + claritySd * kLogisticUnit * logisticOf(U(base + 40u + j)), clarity[j], isCycle);
      }
      var budget = minutes - assessMinutes;
      var reserve = select(f32(0), min(reteachMinutes, budget / f32(chunks + 1u)), checking);
      budget = budget - reserve;
      var reteaches = f32(0);
      var missed = f32(0);
      var lastMet = f32(0);
      var mean = f32(0);
      for (var j = 0u; j < chunks; j = j + 1u) {
        let b = difficulty - f32(0.3) * f32(chunks - 1u - j);
        let given = budget / f32(chunks - j);
        budget = budget - given;
        var extra = f32(0);
        for (var attempt = 0u; attempt < 2u; attempt = attempt + 1u) {
          if (attempt == 1u && extra <= f32(0)) { break; }
          let relative = (given + select(f32(0), extra, attempt == 1u)) / nominal;
          let support = (f32(1) - exp(f32(-2) * relative)) / (f32(1) - exp(f32(-2)));
          mean = f32(0);
          for (var i = 0u; i < n; i = i + 1u) {
            let task = atLogit(theta[i] + c[j] + prerequisite * (prev[i] - f32(1)) - b, kA);
            m[i] = chance(shift(task, scaffold * support));
            mean = mean + m[i] / f32(n);
          }
          if (attempt == 0u && checking) {
            let estimate = mean + select(f32(0), f32(0.15), informal) + select(f32(0.04), f32(0.10), informal) * kLogisticUnit * logisticOf(U(base + 60u + j));
            if (estimate >= reteachThreshold && mean < reteachThreshold) { missed = missed + f32(1); }
            if (estimate < reteachThreshold) {
              if (reserve > f32(0)) {
                extra = reserve;
                reserve = f32(0);
              } else {
                extra = min(reteachMinutes, budget * f32(0.5));
                budget = budget - extra;
              }
              if (extra > f32(0)) {
                c[j] = c[j] + reteachClarity * extra / reteachMinutes;
                reteaches = reteaches + f32(1);
              }
            } else if (j == chunks - 1u && reserve > f32(0)) {
              extra = reserve;
            }
          }
        }
        var met = f32(0);
        for (var i = 0u; i < n; i = i + 1u) {
          met = met + select(f32(0), f32(1), m[i] >= threshold) / f32(n);
          prev[i] = m[i];
        }
        chunkMet[j] = chunkMet[j] + met / f32(classes);
        lastMet = met;
        if (!isCycle) { stat(j, mean); }
      }
      let outcomeMet = lastMet;
      var estimated = f32(0.85);
      if (summative) {
        estimated = f32(0);
        for (var i = 0u; i < n; i = i + 1u) {
          estimated = estimated + select(f32(0), f32(1), m[i] + f32(0.12) * kLogisticUnit * logisticOf(U(base + 100u + i)) >= threshold) / f32(n);
        }
      } else if (checking && !informal) {
        estimated = clamp(lastMet + f32(0.04) * kLogisticUnit * logisticOf(U(base + 200u)), f32(0), f32(1));
      } else if (informal) {
        estimated = clamp(lastMet + f32(0.15) + f32(0.10) * kLogisticUnit * logisticOf(U(base + 200u)), f32(0), f32(1));
      }
      if (!isCycle) {
        stat(8u, mean);
        stat(9u, outcomeMet);
        stat(10u, reteaches);
        stat(11u, missed);
        stat(12u, assessMinutes);
        stat(13u, abs(estimated - outcomeMet));
      }
      metSum = metSum + outcomeMet / f32(classes);
      masterySum = masterySum + mean / f32(classes);
    }
    if (isCycle) {
      var claritySum = f32(0);
      for (var j = 0u; j < chunks; j = j + 1u) { claritySum = claritySum + clarity[j] / f32(chunks); }
      stat(o, metSum);
      stat(16u + o, masterySum);
      stat(32u + o, difficulty);
      stat(48u + o, claritySum);
      let k0 = 900000u + o * 16u;
      var estMet = f32(-1);
      if (evidence > f32(0.5)) {
        ARR(measured, f32, 8);
        for (var j = 0u; j < chunks; j = j + 1u) {
          let z = kLogisticUnit * logisticOf(U(k0 + j));
          var x = chunkMet[j] + f32(0.03) * z;
          if (evidence > f32(1.5) && evidence < f32(2.5)) { x = chunkMet[j] + f32(0.6) * (f32(0.8) - chunkMet[j]) + f32(0.05) + f32(0.03) * z; }
          if (evidence < f32(1.5)) { x = f32(0.55) + f32(0.35) * masterySum + f32(0.04) * z; }
          measured[j] = x;
        }
        let z = kLogisticUnit * logisticOf(U(k0 + 15u));
        estMet = metSum + f32(0.03) * z;
        if (evidence > f32(1.5) && evidence < f32(2.5)) { estMet = clamp(metSum + f32(0.6) * (f32(0.75) - metSum) + f32(0.05) + f32(0.03) * z, f32(0), f32(1)); }
        if (evidence < f32(1.5)) { estMet = clamp(f32(0.55) + f32(0.4) * metSum + f32(0.05) * z, f32(0), f32(1)); }
        for (var j = 0u; j < chunks; j = j + 1u) {
          if (measured[j] < P(15u)) { clarity[j] = clarity[j] + P(14u) * clamp((f32(0.85) - chunkMet[j]) / f32(0.3), f32(0), f32(1)); }
        }
        if (estMet > P(16u)) { difficulty = difficulty + P(18u); }
        else if (estMet < P(17u)) { difficulty = difficulty - P(18u); }
      }
      stat(64u + o, estMet);
    }
  }
)

namespace syn::models {

const kernel::Model* assessmentSessionModel() {
  static const kernel::Model model = [] {
    kernel::Model m;
    m.name = "assessmentSession";
    m.title = "Formative and summative assessment in a session; the assessment cycle over offerings";
    m.maxStats = 80;
    SYN_USE_AGENT(m, session_agent);
    m.compile = [](const Value& spec) {
      const Value& c = spec.sub("assessment");
      const int size = std::max(2, std::min(32, c.integer("classSize", 25)));
      const int chunks = std::max(2, std::min(8, c.integer("chunks", 4)));
      std::vector<double> params = {double(size), double(chunks), c.num("minutes", 45.0), c.num("betweenClassSd", 0.5), c.num("withinClassSd", 0.9),
                                    c.num("clarityMean", 0.0), c.num("claritySd", 0.8), c.num("scaffold", 1.5), c.num("prerequisite", 1.5),
                                    c.num("difficulty", 0.0), c.num("reteachThreshold", 0.7), c.num("reteachMinutes", 4.0), c.num("reteachClarity", 1.0),
                                    c.num("threshold", 0.6), c.num("revise", 0.5), c.num("gapThreshold", 0.7), c.num("raiseAbove", 0.85),
                                    c.num("lowerBelow", 0.4), c.num("step", 0.3), 0.0, double(std::min(16, c.integer("offerings", 8))),
                                    double(c.integer("classesPerOffering", 8)), 0.0, 0.0};
      auto clarity = c.nums("chunkClarity", {0.4, -1.2, 0.3, -0.5});
      clarity.resize(8, 0.0);
      params.insert(params.end(), clarity.begin(), clarity.end());
      std::vector<std::vector<double>> sessions, cycle;
      for (double f : {0.0, 1.0, 2.0})
        for (double s : {0.0, 1.0}) sessions.push_back({0, f, s});
      for (double e : {0.0, 1.0, 2.0, 3.0}) cycle.push_back({1, e, 0});
      return std::vector<Plan>{makePlan("assessmentSession", static_cast<std::uint32_t>(c.integer("classes", 300)), 14, params, sessions, seedOf(spec)),
                               makePlan("assessmentSession", static_cast<std::uint32_t>(c.integer("instructors", 64)), 80, params, cycle, seedOf(spec))};
    };
    m.summarise = [](const Value&, const std::vector<Plan>& plans, const std::vector<Output>& out) {
      const Plan& p = plans[0];
      const int chunks = static_cast<int>(p.params[1]);
      const char* checks[] = {"none", "informal check", "formal check"};
      Value sessions = Value::array();
      for (std::uint32_t j = 0; j < p.jobs; ++j) {
        const int f = static_cast<int>(p.jobRows[j * p.jobStride + 1]);
        const bool s = p.jobRows[j * p.jobStride + 2] > 0.5;
        sessions.push(Value::object()
                          .set("name", std::string(checks[f]) + (s ? " + summative" : ""))
                          .set("chunkMastery", Value::Numbers(out[0].range(p, j, 0, chunks)))
                          .set("finalMastery", out[0].mean(p, j, 8))
                          .set("outcomeMet", out[0].mean(p, j, 9))
                          .set("reteaches", out[0].mean(p, j, 10))
                          .set("missedGaps", out[0].mean(p, j, 11))
                          .set("assessmentMinutes", out[0].mean(p, j, 12))
                          .set("estimateError", out[0].mean(p, j, 13)));
      }
      const Plan& q = plans[1];
      const int offerings = static_cast<int>(q.params[20]);
      const char* evidence[] = {"no assessment", "grades", "self-report (indirect)", "direct assessment"};
      Value cycle = Value::array();
      for (std::uint32_t j = 0; j < q.jobs; ++j)
        cycle.push(Value::object()
                       .set("evidence", evidence[j])
                       .set("met", Value::Numbers(out[1].range(q, j, 0, offerings)))
                       .set("mastery", Value::Numbers(out[1].range(q, j, 16, offerings)))
                       .set("difficulty", Value::Numbers(out[1].range(q, j, 32, offerings)))
                       .set("clarity", Value::Numbers(out[1].range(q, j, 48, offerings)))
                       .set("estimatedMet", Value::Numbers(out[1].range(q, j, 64, offerings))));
      return Value::object().set("sessions", sessions).set("cycle", cycle);
    };
    return m;
  }();
  return &model;
}

}  // namespace syn::models
