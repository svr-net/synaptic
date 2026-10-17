// Professional development (chapter 14) in the space: lifelong learning over three years of
// teaching.
//
// An agent is an instruction librarian. Their teaching is a point of the cognitive channel;
// their self-efficacy a point of the affective channel; their expert blind spot — how far they
// have forgotten what is hard for a novice (search strategies, jargon, evaluation criteria) — a
// point of the cognitive channel that every month of expertise turns towards them. What a month
// of sessions teaches is the chance of their teaching less the blind spot. New ideas arrive each
// month from their own experience and, when they have them, from a community of practice (shared
// lesson plans, a journal club), from watching colleagues teach, from continuing education
// (courses, webinars, a certificate) and from the professional literature. An idea fits their
// own style with a uniform chance: copied as it is, an idea that does not fit (a colleague's
// humour) makes the teaching worse; adapted, it keeps what fits. Hard sessions lower efficacy,
// a community's support raises it, and a drained librarian stops trying new things.
// Becoming a student again (Brookfield learning to swim) turns the blind spot back. Jobs:
// alone, with each source, copying or adapting, and all of them.
#include <string>

#include "syn/models/models.hpp"

// The agent program, in WGSL.
constexpr const char* growth_agent = R"wgsl(
  let community = J(0u) > f32(0.5);
  let observe = u32(J(1u));
  let courses = J(2u) > f32(0.5);
  let literature = J(3u) > f32(0.5);
  let student = J(4u) > f32(0.5);
  let months = u32(P(0u));
  var teaching = sampleLevel(U(0u), P(1u), P(2u), kA);
  var efficacy = atChance(P(3u), kB);
  var blindSpot = atChance(P(4u), kA);
  let start = logit(teaching);
  var adopted = f32(0);
  var backfired = f32(0);
  var drainedMonths = f32(0);
  for (var mo = 0u; mo < months; mo = mo + 1u) {
    let base = 10u + mo * 32u;
    let reaches = chance(teaching) * (f32(1) - P(5u) * chance(blindSpot));
    stat(mo, reaches);
    // Expertise grows, and with it the distance from the novice.
    blindSpot = raise(blindSpot, P(6u));
    if (student && mo % 6u == 5u) { blindSpot = lower(blindSpot, P(7u)); }
    // Hard sessions; the support of a community.
    if (U(base) < P(8u)) { efficacy = lower(efficacy, P(9u)); }
    if (community) { efficacy = raise(efficacy, P(10u)); }
    efficacy = raise(efficacy, P(11u) * reaches);
    let drained = !inside(efficacy, chanceCap(P(12u)));
    if (drained) {
      drainedMonths = drainedMonths + f32(1);
      continue;
    }
    // Ideas this month.
    var rate = P(13u);
    if (community) { rate = rate + P(14u); }
    if (observe > 0u) { rate = rate + P(15u); }
    if (courses) { rate = rate + P(16u); }
    if (literature) { rate = rate + P(17u); }
    if (U(base + 1u) < rate) {
      let fit = U(base + 2u);
      // Copied, it works as well as it fits; adapted, the misfit is worked out.
      let copied = observe == 1u && U(base + 3u) < P(15u) / rate;
      let effect = select(P(18u) * (f32(0.5) + f32(0.5) * fit), P(19u) * (fit - P(20u)), copied);
      teaching = shift(teaching, effect);
      adopted = adopted + f32(1);
      if (effect < f32(0)) { backfired = backfired + f32(1); }
    }
  }
  stat(40u, logit(teaching) - start);
  stat(41u, chance(efficacy));
  stat(42u, chance(blindSpot));
  stat(43u, adopted);
  stat(44u, backfired);
  stat(45u, drainedMonths);
)wgsl";

namespace syn::models {

const kernel::Model* professionalGrowthModel() {
  static const kernel::Model model = [] {
    kernel::Model m;
    m.name = "professionalGrowth";
    m.title = "Professional development: communities of practice, observing colleagues, continuing education, the literature, becoming a student again";
    m.maxStats = 46;
    m.agentSource = growth_agent;
    m.compile = [](const Value& spec) {
      const Value& c = spec.sub("professionalGrowth");
      const int months = std::max(6, std::min(40, c.integer("months", 36)));
      const std::vector<double> params = {double(months), c.num("teaching", 0.3), c.num("teachingSd", 0.6), c.num("efficacy", 0.65),
                                          c.num("blindSpot", 0.1), c.num("blindCost", 0.25), c.num("expertise", 0.03), c.num("beginnerAgain", 0.5),
                                          c.num("hardSessions", 0.35), c.num("hardCost", 0.08), c.num("support", 0.06), c.num("success", 0.035),
                                          c.num("drained", 0.4), c.num("ownIdeas", 0.05), c.num("communityIdeas", 0.12),
                                          c.num("observingIdeas", 0.12), c.num("courseIdeas", 0.1), c.num("literatureIdeas", 0.08),
                                          c.num("adapted", 0.12), c.num("copied", 0.3), c.num("misfit", 0.45)};
      const std::vector<std::vector<double>> rows = {{0, 0, 0, 0, 0}, {1, 0, 0, 0, 0}, {0, 1, 0, 0, 0}, {0, 2, 0, 0, 0}, {0, 0, 1, 0, 0},
                                                     {0, 0, 0, 1, 0}, {0, 0, 0, 0, 1}, {1, 2, 1, 1, 1}};
      return std::vector<Plan>{makePlan("professionalGrowth", static_cast<std::uint32_t>(c.integer("librarians", 1000)), 46, params, rows, seedOf(spec))};
    };
    m.summarise = [](const Value&, const std::vector<Plan>& plans, const std::vector<Output>& out) {
      const Plan& p = plans[0];
      const Output& o = out[0];
      const auto months = static_cast<std::uint32_t>(p.params[0]);
      const char* names[] = {"on your own", "a community of practice", "observing colleagues, copying", "observing colleagues, adapting",
                             "continuing education", "the professional literature", "becoming a student again", "all of them"};
      Value designs = Value::array();
      for (std::uint32_t j = 0; j < p.jobs; ++j)
        designs.push(Value::object()
                         .set("name", names[j])
                         .set("reach", Value::Numbers(o.range(p, j, 0, months)))
                         .set("gain", o.mean(p, j, 40))
                         .set("efficacy", o.mean(p, j, 41))
                         .set("blindSpot", o.mean(p, j, 42))
                         .set("adopted", o.mean(p, j, 43))
                         .set("backfired", o.mean(p, j, 44))
                         .set("drainedMonths", o.mean(p, j, 45)));
      return Value::object()
          .set("designs", designs)
          .set("months", static_cast<double>(months))
          .set("outlets", Value::array()
                              .push(Value::object().set("kind", "Associations").set("examples", "ALA, ACRL (Instruction Section, Framework Sandbox), AASL, PLA, LOEX, LIRT; regional associations"))
                              .push(Value::object().set("kind", "Continuing education").set("examples", "WebJunction, Infopeople, Library Juice Academy, ACRL Immersion, LIS programs, edX"))
                              .push(Value::object().set("kind", "Literature").set("examples", "Communications in Information Literacy, C&RL, C&RL News, Knowledge Quest, School Library Research, ERIC"))
                              .push(Value::object().set("kind", "Communities").set("examples", "ILI-L listserv, journal clubs, teaching and learning centres, social media")));
    };
    return m;
  }();
  return &model;
}

}  // namespace syn::models
