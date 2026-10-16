// Motivation (Svinicki's expectancy × value) and mindset (Dweck) in the space.
//
// Expectancy × value: value is a point of the affective channel nudged by relevance, choice and
// the interest of a well-pitched challenge; self-efficacy (the expectancy) is a share; motivation
// is the chance that both hold, value × efficacy. Success is the cap test of the point dilated by
// effort and set-up against the task's offset; self-efficacy follows experience by a mixture
// towards the perceived outcome (blend), and guided self-assessment shrinks the perception error.
// Ability is a level dilated by each week's effort.
// Mindset: a learner's growth mindset is a share; praising ability lowers it after successes,
// "not yet" raises it after failures; growth-minded learners seek challenge, keep their effort
// and switch to better strategies after failure, fixed-minded ones withdraw effort and may quit.
// Skill is a level dilated by effort × strategy quality × stretch.
// Plans: the term under no levers, each lever and all five; mindset under three feedback styles.
#include <string>

#include "syn/models/models.hpp"
#include "syn/kernel/agent.hpp"

SYN_AGENT(motivation_agent,
  if (J(0u) < f32(0.5)) {
    let weeks = u32(P(0u));
    let relevance = J(1u);
    let fit = J(2u);
    let choice = J(3u);
    let setup = J(4u);
    let assess = J(5u);
    var ability = sampleLevel(U(0u), P(4u), P(5u), kA);
    let start = logit(ability);
    var efficacy = sampleShare(U(1u), P(2u) + f32(0.1) * choice, P(3u), kB);
    var bias = P(6u) * kLogisticUnit * logisticOf(U(2u));
    var engaged = true;
    var weak = 0u;
    for (var t = 0u; t < weeks; t = t + 1u) {
      let miss = kLogisticUnit * logisticOf(U(10u + 3u * t));
      let offset = f32(0.6) + (f32(1) - fit) * f32(1.6) * miss;
      let interest = f32(0.2) - f32(0.12) * (offset - f32(0.6)) * (offset - f32(0.6));
      let value = nudge(atChance(P(7u), kB), f32(0.3) * relevance + f32(0.12) * choice + interest);
      let m = select(f32(0), chance(value) * chance(efficacy), engaged);
      weak = select(0u, weak + 1u, m < P(8u));
      if (weak >= 3u) { engaged = false; }
      let effort = select(f32(0), m, engaged);
      let attempt = shift(equator(kA), f32(1.6) * effort + f32(0.5) * setup - offset);
      let success = inside(attempt, chanceCap(U(11u + 3u * t)));
      if (engaged) {
        ability = shift(ability, f32(0.25) * effort * (f32(1) + f32(0.3) * setup));
        let perceived = nudge(atChance(select(f32(0), f32(1), success), kB), f32(0.5) * bias);
        efficacy = blend(efficacy, perceived, f32(0.12) + f32(0.12) * setup);
        bias = bias * (f32(1) - f32(0.15) * assess);
      }
      stat(t, m);
      stat(16u + t, chance(value));
      stat(32u + t, chance(efficacy));
      stat(48u + t, logit(ability) - start);
      stat(64u + t, abs(bias));
      stat(80u + t, f32(engaged));
    }
    stat(96u, logit(ability) - start);
  } else {
    let weeks = u32(P(9u));
    let style = J(1u);
    let coaching = clamp(J(2u), f32(0), f32(1));
    let standards = J(3u) > f32(0.5);
    var growth = sampleShare(U(0u), P(10u), P(11u), kB);
    let startsGrowth = chance(growth) >= f32(0.5);
    ARR(quality, f32, 4);
    for (var k = 0u; k < 4u; k = k + 1u) { quality[k] = f32(0.2) + f32(0.8) * U(1u + k); }
    var strategy = 0u;
    var skill = atLogit(f32(0), kA);
    var effort = atChance(f32(0.6), kB);
    var engaged = true;
    for (var t = 0u; t < weeks; t = t + 1u) {
      let g = chance(growth);
      let pick = min(3u, u32(floor(U(10u + 5u * t) * f32(4))));
      let challenge = U(11u + 5u * t) < f32(0.15) + f32(0.7) * g;
      if (engaged) {
        let difficulty = select(f32(-0.5), f32(1), challenge) - select(f32(0.8), f32(0), standards);
        let attempt = shift(equator(kA), f32(1.5) * chance(effort) - difficulty);
        let ok = inside(attempt, chanceCap(U(12u + 5u * t)));
        let stretch = max(f32(0.2), select(f32(0.6), f32(1.6), challenge) - select(f32(0.3), f32(0), standards));
        skill = shift(skill, P(12u) * chance(effort) * quality[strategy] * stretch);
        if (ok) {
          if (style < f32(0.5)) { growth = lower(growth, f32(0.03)); }
          if (style > f32(1.5)) { growth = raise(growth, f32(0.02)); }
        } else {
          if (style > f32(1.5)) { growth = raise(growth, f32(0.04)); }
          let g2 = chance(growth);
          effort = nudge(effort, f32(0.12) * g2 - f32(0.18) * (f32(1) - g2));
          if (U(13u + 5u * t) < f32(0.6) * g2 && quality[pick] >= quality[strategy] - f32(0.1)) { strategy = pick; }
          if (chance(effort) < f32(0.2) && U(14u + 5u * t) < f32(0.5) * (f32(1) - g2)) { engaged = false; }
        }
        growth = raise(growth, f32(0.04) * coaching);
      }
      stat(t, f32(engaged));
      stat(32u + t, chance(growth));
      stat(64u + t, f32(engaged && challenge));
      stat(96u + t, select(f32(0), logit(skill), startsGrowth));
      stat(128u + t, select(logit(skill), f32(0), startsGrowth));
    }
    stat(160u, f32(startsGrowth));
    stat(161u, logit(skill));
  }
)

namespace syn::models {

const kernel::Model* motivationModel() {
  static const kernel::Model model = [] {
    kernel::Model m;
    m.name = "motivation";
    m.title = "Expectancy × value with five levers; growth and fixed mindsets under three kinds of feedback";
    m.maxStats = 162;
    SYN_USE_AGENT(m, motivation_agent);
    m.compile = [](const Value& spec) {
      const Value& c = spec.sub("motivation");
      const int weeks = std::min(16, c.integer("weeks", 15)), mweeks = std::min(32, c.integer("mindsetWeeks", 30));
      const std::vector<double> params = {double(weeks), 0.0, 0.5, 0.15, 0.0, 1.0, 0.25, 0.45, 0.08, double(mweeks), 0.5, 0.25, 0.12};
      std::vector<std::vector<double>> terms = {{0, 0, 0, 0, 0, 0}};
      for (int k = 0; k < 5; ++k) {
        std::vector<double> r(6, 0.0);
        r[1 + k] = 1.0;
        terms.push_back(r);
      }
      terms.push_back({0, 1, 1, 1, 1, 1});
      std::vector<std::vector<double>> minds;
      for (int f = 0; f < 3; ++f) minds.push_back({1, double(f), c.num("coaching", 0.0), c.flag("highStandards", true) ? 1.0 : 0.0, 0, 0});
      const std::uint32_t learners = static_cast<std::uint32_t>(c.integer("learners", 400));
      return std::vector<Plan>{makePlan("motivation", learners, 97, params, terms, seedOf(spec)),
                               makePlan("motivation", learners, 162, params, minds, seedOf(spec))};
    };
    m.summarise = [](const Value&, const std::vector<Plan>& plans, const std::vector<Output>& out) {
      const Plan& p = plans[0];
      const Output& o = out[0];
      const int weeks = static_cast<int>(p.params[0]);
      const char* labels[] = {"no levers", "emphasise relevance", "appropriate challenge", "choice and control", "set up for success",
                              "guided self-assessment", "all five"};
      Value terms = Value::array();
      for (std::uint32_t j = 0; j < p.jobs; ++j) {
        const auto motivation = o.range(p, j, 0, weeks);
        terms.push(Value::object()
                       .set("label", labels[j])
                       .set("motivation", Value::Numbers(motivation))
                       .set("value", Value::Numbers(o.range(p, j, 16, weeks)))
                       .set("efficacy", Value::Numbers(o.range(p, j, 32, weeks)))
                       .set("ability", Value::Numbers(o.range(p, j, 48, weeks)))
                       .set("calibration", Value::Numbers(o.range(p, j, 64, weeks)))
                       .set("engaged", Value::Numbers(o.range(p, j, 80, weeks)))
                       .set("gain", o.mean(p, j, 96))
                       .set("finalMotivation", motivation.back()));
      }
      Value r = Value::object();
      r.set("terms", terms);
      std::vector<double> offset = grid(-3, 4, 71), interest;
      for (double x : offset) interest.push_back(0.2 - 0.12 * (x - 0.6) * (x - 0.6));
      r.set("offset", Value::Numbers(offset)).set("interest", Value::Numbers(interest));
      const Plan& q = plans[1];
      const Output& oq = out[1];
      const int mweeks = static_cast<int>(q.params[9]);
      const char* styles[] = {"praise ability", "grades only", "praise process, \"not yet\""};
      Value minds = Value::array();
      for (std::uint32_t j = 0; j < q.jobs; ++j) {
        const double growthShare = oq.mean(q, j, 160);
        std::vector<double> fixed, grow;
        for (int t = 0; t < mweeks; ++t) {
          grow.push_back(growthShare > 0 ? oq.mean(q, j, 96 + t) / growthShare : 0.0);
          fixed.push_back(growthShare < 1 ? oq.mean(q, j, 128 + t) / (1.0 - growthShare) : 0.0);
        }
        const auto persisting = oq.range(q, j, 0, mweeks);
        minds.push(Value::object()
                       .set("feedback", styles[j])
                       .set("skillFixed", Value::Numbers(fixed))
                       .set("skillGrowth", Value::Numbers(grow))
                       .set("persisting", Value::Numbers(persisting))
                       .set("mindset", Value::Numbers(oq.range(q, j, 32, mweeks)))
                       .set("challenge", Value::Numbers(oq.range(q, j, 64, mweeks)))
                       .set("gain", oq.mean(q, j, 161))
                       .set("persist", persisting.back()));
      }
      r.set("mindset", minds);
      return r;
    };
    return m;
  }();
  return &model;
}

}  // namespace syn::models
