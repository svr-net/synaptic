// Maslow's hierarchy of needs in the space.
//
// Each need is a point of the affective channel whose chance is how far the need is met. A level
// motivates only once the levels below it are met: the gate of a need is the equator point
// dilated by (met − threshold)/width, read as a chance, and the share of a level's motivating
// potential that is available is the product of the gates below it. The humanist teacher's
// climate raises safety, belonging and esteem towards the learner: raise(N, lever). The dominant
// need is the lowest one outside the cap of the threshold.
// Jobs: 41 levels of safety and belonging (readiness with and without a supportive climate), then
// one student (as specified) without and with the climate.
#include "syn/models/models.hpp"
#include "syn/kernel/agent.hpp"

SYN_AGENT(maslow_agent,
  let threshold = P(0u);
  let width = P(1u);
  let climate = J(6u);
  ARR(need, V5, 5);
  for (var k = 0u; k < 5u; k = k + 1u) { need[k] = atChance(J(k), kB); }
  need[1] = raise(need[1], climate * J(7u));
  need[2] = raise(need[2], climate * J(8u));
  need[3] = raise(need[3], climate * J(9u));
  var available = f32(1);
  var dominant = f32(5);
  for (var k = 0u; k < 5u; k = k + 1u) {
    stat(k, chance(need[k]));
    stat(5u + k, available);
    let gate = chance(shift(equator(kB), (chance(need[k]) - threshold) / width));
    available = available * gate;
    if (dominant > f32(4.5) && !inside(need[k], chanceCap(threshold))) { dominant = f32(k); }
  }
  stat(10u, dominant);
)

namespace syn::models {

const kernel::Model* maslowModel() {
  static const kernel::Model model = [] {
    kernel::Model m;
    m.name = "maslow";
    m.title = "Maslow's prepotent needs: gates, readiness and a supportive classroom climate";
    m.maxStats = 11;
    SYN_USE_AGENT(m, maslow_agent);
    m.compile = [](const Value& spec) {
      const Value& c = spec.sub("humanism");
      std::vector<std::vector<double>> rows;
      const auto g = grid(0, 1, 41);
      for (double s : g) rows.push_back({0.9, s, s, 0.45, 0.2, 0, 0, 0, 0, 0});
      for (double s : g) rows.push_back({0.9, s, s, 0.45, 0.2, 0, 1, 0.6, 0.6, 0.4});
      const std::vector<double> student = {c.num("physiological", 0.9), c.num("safety", 0.5), c.num("belonging", 0.45), c.num("esteem", 0.4), 0.2, 0};
      for (double climate : {0.0, 1.0}) {
        auto r = student;
        r.insert(r.end(), {climate, c.num("empathy", 0.7), c.num("belongingSupport", 0.6), c.num("recognition", 0.5)});
        rows.push_back(r);
      }
      return std::vector<Plan>{makePlan("maslow", 1, 11, {c.num("threshold", 0.55), c.num("width", 0.07)}, rows, seedOf(spec))};
    };
    m.summarise = [](const Value&, const std::vector<Plan>& plans, const std::vector<Output>& out) {
      const Plan& p = plans[0];
      const Output& o = out[0];
      const char* names[] = {"physiological", "safety", "belonging and love", "esteem", "self-actualisation"};
      Value r = Value::object();
      std::vector<double> g = grid(0, 1, 41), plain, climate;
      for (std::uint32_t j = 0; j < 41; ++j) {
        plain.push_back(o.mean(p, j, 5 + 3));  // readiness: the gate product through belonging
        climate.push_back(o.mean(p, 41 + j, 5 + 3));
      }
      r.set("needGrid", Value::Numbers(g)).set("readiness", Value::Numbers(plain)).set("readinessWithClimate", Value::Numbers(climate));
      r.set("student", Value::Numbers(o.range(p, 82, 0, 5))).set("supported", Value::Numbers(o.range(p, 83, 0, 5)));
      r.set("activation", Value::Numbers(o.range(p, 82, 5, 5))).set("activationSupported", Value::Numbers(o.range(p, 83, 5, 5)));
      r.set("dominant", names[std::min(4, static_cast<int>(std::lround(o.mean(p, 82, 10))))]);
      r.set("dominantSupported", names[std::min(4, static_cast<int>(std::lround(o.mean(p, 83, 10))))]);
      r.set("needs", std::vector<std::string>(names, names + 5));
      return r;
    };
    return m;
  }();
  return &model;
}

}  // namespace syn::models
