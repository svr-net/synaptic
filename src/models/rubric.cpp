// Rubrics and self-assessment (chapter 9) in the space.
//
// Rubrics: a piece of work has a true quality, a level; four instructors rate it. A rater sees the
// quality dilated by their severity plus noise and places it against their own reading of the
// level boundaries: with no rubric, severities, noise and boundaries vary a lot; a vague rubric
// ("some", "often") leaves the boundaries to the rater; a quantified one counts what is countable
// (references), reliably and invalidly; a precise one describes the levels clearly. Shared with
// learners, a useful rubric raises the quality of the work, a count-based one adds references.
// The raters' severities and boundaries are the same for every work (compiled into the job), and
// the kernel accumulates the agreement, the marginals for Cohen's κ and the sums behind each
// rater's validity.
// Self-assessment: a learner's skill is a share; their self-assessment is the point of their skill
// mixed towards a flattering 0.6 (by β, more for the least skilled) and nudged by flattery and
// noise. Reflection raises skill a little each round; feedback on the self-assessment shrinks β
// and the flattery and doubles the gain from reflection.
#include <algorithm>
#include <string>

#include "syn/models/models.hpp"
#include "syn/kernel/agent.hpp"

SYN_AGENT(rubric_agent,
  if (J(0u) < f32(0.5)) {
    let kind = J(1u);
    let sharedRubric = J(2u) > f32(0.5);
    let levels = u32(J(3u));
    let noise = J(4u);
    let qualityGain = J(5u);
    let countGain = J(6u);
    let quality = shift(sampleLevel(U(0u), f32(0), f32(1), kA), select(f32(0), qualityGain, sharedRubric));
    let q = logit(quality);
    let count = max(f32(0), floor(f32(4) + f32(1.2) * (f32(0.35) * q + f32(0.94) * kLogisticUnit * logisticOf(U(1u))) + select(f32(0), countGain, sharedRubric) + f32(0.5)));
    ARR(level, u32, 4);
    for (var r = 0u; r < 4u; r = r + 1u) {
      let z = kLogisticUnit * logisticOf(U(2u + r));
      var seen = shift(quality, J(8u + r) + noise * z);
      if (kind > f32(1.5) && kind < f32(2.5)) { seen = atLogit((count - f32(4)) / f32(1.2) + noise * z, kA); }
      var lv = 0u;
      for (var k = 0u; k + 1u < levels; k = k + 1u) {
        if (logit(seen) > J(12u + r * 9u + k)) { lv = lv + 1u; }
      }
      level[r] = lv;
      stat(6u + r * 10u + lv, f32(1));
      stat(46u + r, f32(lv));
      stat(50u + r, f32(lv) * f32(lv));
      stat(54u + r, f32(lv) * q);
    }
    var pair = 0u;
    for (var a = 0u; a < 4u; a = a + 1u) {
      for (var b = a + 1u; b < 4u; b = b + 1u) {
        stat(pair, f32(level[a] == level[b]));
        pair = pair + 1u;
      }
    }
    stat(58u, q);
    stat(59u, q * q);
    stat(60u, count);
  } else {
    let feedback = J(1u) > f32(0.5);
    let rounds = u32(P(0u));
    let u0 = U(0u);
    let quartile = min(3u, u32(floor(u0 * f32(4))));
    var skill = sampleLevel(u0, f32(0), f32(1.2), kA);
    var beta = P(1u);
    var flattery = P(2u);
    stat(80u + quartile, f32(1));
    for (var r = 0u; r < rounds; r = r + 1u) {
      let a = chance(skill);
      let perceived = chance(nudge(blend(skill, atChance(f32(0.6), kA), beta), flattery + P(3u) * kLogisticUnit * logisticOf(U(1u + r))));
      stat(r * 8u + quartile, a);
      stat(r * 8u + 4u + quartile, perceived);
      stat(64u + r, abs(perceived - a));
      stat(72u + r, a);
      skill = raise(skill, P(5u) * select(f32(1), f32(2), feedback));
      if (feedback) {
        beta = beta * (f32(1) - P(4u));
        flattery = flattery * (f32(1) - P(4u));
      }
    }
  }
)

namespace syn::models {

namespace {
const char* kKinds[4] = {"no rubric", "vague rubric", "quantified rubric", "precise rubric"};
// severity sd, noise, boundary jitter, quality gain when shared, references added when shared
const double kRaters[4][5] = {{0.5, 0.6, 0.4, 0.0, 0.0}, {0.3, 0.45, 0.35, 0.1, 0.0}, {0.0, 0.05, 0.0, 0.0, 1.5}, {0.1, 0.25, 0.08, 0.3, 0.0}};

// Job row: kind 0, rubric kind, shared, levels, noise, quality gain, count gain, 0, severities (4),
// boundaries (4 raters × 9).
std::vector<double> rubricRow(int kind, bool shared, int levels, std::uint32_t seed) {
  std::vector<double> r = {0, double(kind), shared ? 1.0 : 0.0, double(levels), kRaters[kind][1], kRaters[kind][3], kRaters[kind][4], 0};
  for (std::uint32_t k = 0; k < 4; ++k) r.push_back(kRaters[kind][0] * logisticDraw(seed ^ 0xBA7Eu, k, 0));
  for (std::uint32_t k = 0; k < 4; ++k) {
    std::vector<double> b;
    for (int j = 1; j < levels; ++j) b.push_back(-1.2 + 2.4 * j / levels + kRaters[kind][2] * logisticDraw(seed ^ 0xBA7Eu, k, 1 + j));
    std::sort(b.begin(), b.end());
    b.resize(9, 1e6);
    r.insert(r.end(), b.begin(), b.end());
  }
  return r;
}
}  // namespace

const kernel::Model* rubricModel() {
  static const kernel::Model model = [] {
    kernel::Model m;
    m.name = "rubric";
    m.title = "Rubrics: rater agreement and validity; self-assessment and feedback";
    m.maxStats = 84;
    SYN_USE_AGENT(m, rubric_agent);
    m.compile = [](const Value& spec) {
      const Value& c = spec.sub("rubric");
      const std::uint32_t seed = seedOf(spec);
      std::vector<std::vector<double>> rows;
      for (int kind = 0; kind < 4; ++kind)
        for (bool shared : {false, true}) rows.push_back(rubricRow(kind, shared, 3, seed));
      for (int levels = 2; levels <= 6; ++levels) rows.push_back(rubricRow(3, false, levels, seed));
      std::vector<std::vector<double>> self;
      for (double f : {0.0, 1.0}) {
        std::vector<double> r(rows[0].size(), 0.0);
        r[0] = 1;
        r[1] = f;
        self.push_back(r);
      }
      const std::vector<double> params = {double(std::min(8, c.integer("rounds", 4))), c.num("regression", 0.6), c.num("flattery", 0.05), c.num("noise", 0.08),
                                          c.num("feedbackShrink", 0.45), c.num("reflection", 0.03)};
      return std::vector<Plan>{makePlan("rubric", static_cast<std::uint32_t>(c.integer("works", 300)), 61, params, rows, seed),
                               makePlan("rubric", static_cast<std::uint32_t>(c.integer("learners", 400)), 84, params, self, seed)};
    };
    m.summarise = [](const Value&, const std::vector<Plan>& plans, const std::vector<Output>& out) {
      const Plan& p = plans[0];
      const Output& o = out[0];
      const double W = p.agents;
      auto rubric = [&](std::uint32_t j) {
        const int levels = static_cast<int>(p.jobRows[j * p.jobStride + 3]);
        double agree = 0, kappa = 0;
        int pair = 0;
        for (int a = 0; a < 4; ++a)
          for (int b = a + 1; b < 4; ++b, ++pair) {
            const double po = o.mean(p, j, pair);
            double pe = 0;
            for (int l = 0; l < levels; ++l) pe += o.mean(p, j, 6 + a * 10 + l) * o.mean(p, j, 6 + b * 10 + l);
            agree += po / 6;
            kappa += (pe < 1 ? (po - pe) / (1 - pe) : 1.0) / 6;
          }
        const double mq = o.mean(p, j, 58), vq = o.mean(p, j, 59) - mq * mq;
        double validity = 0;
        for (int r = 0; r < 4; ++r) {
          const double ml = o.mean(p, j, 46 + r), vl = o.mean(p, j, 50 + r) - ml * ml, cov = o.mean(p, j, 54 + r) - ml * mq;
          validity += (vl > 0 && vq > 0 ? cov / std::sqrt(vl * vq) : 0.0) / 4;
        }
        (void)W;
        return Value::object().set("exactAgreement", agree).set("kappa", kappa).set("validity", validity).set("meanQuality", mq).set("meanCount", o.mean(p, j, 60));
      };
      Value rubrics = Value::array();
      for (int kind = 0; kind < 4; ++kind) {
        Value x = rubric(kind * 2);
        rubrics.push(Value::object().set("kind", kKinds[kind]).set("private", x).set("shared", rubric(kind * 2 + 1)));
      }
      std::vector<double> levels, agree, kappa, validity;
      for (std::uint32_t j = 8; j < 13; ++j) {
        const Value x = rubric(j);
        levels.push_back(p.jobRows[j * p.jobStride + 3]);
        agree.push_back(x.num("exactAgreement", 0));
        kappa.push_back(x.num("kappa", 0));
        validity.push_back(x.num("validity", 0));
      }
      Value r = Value::object();
      r.set("rubrics", rubrics);
      r.set("rubricLevels", Value::object().set("levels", Value::Numbers(levels)).set("exactAgreement", Value::Numbers(agree)).set("kappa", Value::Numbers(kappa))
                                .set("validity", Value::Numbers(validity)));
      const Plan& q = plans[1];
      const Output& oq = out[1];
      const int rounds = static_cast<int>(q.params[0]);
      Value self = Value::array();
      for (std::uint32_t j = 0; j < 2; ++j) {
        auto quartiles = [&](int round, int offset) {
          std::vector<double> v;
          for (int k = 0; k < 4; ++k) {
            const double share = oq.mean(q, j, 80 + k);
            v.push_back(share > 0 ? oq.mean(q, j, round * 8 + offset + k) / share : 0.0);
          }
          return Value::Numbers(v);
        };
        self.push(Value::object()
                      .set("feedback", j == 1)
                      .set("actualFirst", quartiles(0, 0))
                      .set("perceivedFirst", quartiles(0, 4))
                      .set("actualLast", quartiles(rounds - 1, 0))
                      .set("perceivedLast", quartiles(rounds - 1, 4))
                      .set("calibrationError", Value::Numbers(oq.range(q, j, 64, rounds)))
                      .set("meanSkill", Value::Numbers(oq.range(q, j, 72, rounds))));
      }
      r.set("selfAssessment", self);
      return r;
    };
    return m;
  }();
  return &model;
}

}  // namespace syn::models
