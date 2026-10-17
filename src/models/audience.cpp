// Identifying audience needs (chapter 7): a one-shot session pitched from what the instructor
// knows, in the space.
//
// An agent is a class. Its learners' prior knowledge are levels of the cognitive channel around
// the class mean; the instructor's belief about the class comes from audience research and
// community data (a prior with a bias and an uncertainty) and, if there is time, a pre-assessment:
// each learner responds when the point of the instrument's participation holds a uniform draw, and
// the response reads their level with the instrument's noise and biases. Only the responses that
// fit the analysis budget are read; the estimate weights prior and responses by their precision.
// The session is pitched at the ZPD's peak: a learner's task is the point of logit k(θ − level),
// the scaffold the dilation shift(T, s), and the gain η·P_assisted·(1 − P_alone) × the share of
// the session left for teaching × engagement (grouping by score costs efficiency). The same class
// pitched on the prior alone gives the value of the information.
// Jobs: instrument × prior, instrument × class size, within-class spread × number of groups.
#include <string>

#include "syn/models/models.hpp"

// The agent program, in WGSL.
constexpr const char* audience_agent = R"wgsl(
  let classMinutes = J(0u);
  let readTime = J(1u);
  let noise = J(2u);
  let bias = J(3u) + J(4u);
  let participation = J(5u);
  let individual = J(6u) > f32(0.5);
  let affective = J(7u);
  let engagement = J(8u);
  let interests = J(9u);
  let instAnxiety = J(10u);
  let priorBias = J(11u);
  let priorSd = J(12u);
  let n = u32(J(13u));
  let within = J(14u);
  let groups = u32(J(15u));
  let community = P(0u);
  let sessionMinutes = P(2u);
  let budget = P(3u);
  let scaffold = P(4u);
  let sharp = P(5u);
  let groupCost = P(6u);
  let anxietyMean = P(7u);
  let rate = P(8u);
  let pHi = P(9u);
  let pLo = P(10u);
  let offset = scaffold / (f32(2) * sharp);
  let sigma2 = within * within;
  let noise2 = noise * noise;
  let reliability = sigma2 / (sigma2 + noise2);
  let m0 = community + priorBias;
  let s02 = priorSd * priorSd;
  let mu = logit(sampleLevel(U(0u), community, P(1u), kA));
  var theta : array<f32, 128>;
  var response : array<f32, 128>;
  var read : array<u32, 128>;
  var nRead = 0u;
  for (var i = 0u; i < n; i = i + 1u) {
    theta[i] = logit(sampleLevel(U(1u + 3u * i), mu, within, kA));
    let responds = inside(atChance(participation, kB), chanceCap(U(2u + 3u * i)));
    response[i] = theta[i] + bias + noise * kLogisticUnit * logisticOf(U(3u + 3u * i));
    if (responds) {
      read[nRead] = i;
      nRead = nRead + 1u;
    }
  }
  var analysis = f32(0);
  if (readTime > f32(0) && nRead > 0u) {
    let canRead = u32(floor(budget / readTime + f32(1e-4)));
    nRead = min(nRead, canRead);
    analysis = readTime * f32(nRead);
  }
  stat(5u, f32(nRead) / f32(n));
  var estimate = m0;
  var mean = f32(0);
  if (nRead > 0u) {
    for (var k = 0u; k < nRead; k = k + 1u) { mean = mean + response[read[k]]; }
    mean = mean / f32(nRead);
    let v = (sigma2 + noise2) / f32(nRead);
    estimate = (m0 / s02 + mean / v) / (f32(1) / s02 + f32(1) / v);
  }
  stat(3u, (estimate - mu) * (estimate - mu));
  stat(4u, abs(estimate - mu));
  var level : array<f32, 128>;
  for (var i = 0u; i < n; i = i + 1u) { level[i] = estimate + offset; }
  var used = 1u;
  if (individual && groups > 1u && nRead >= groups) {
    used = groups;
    for (var a = 1u; a < nRead; a = a + 1u) {
      let key = read[a];
      var b = a;
      while (b > 0u && response[read[b - 1u]] > response[key]) { read[b] = read[b - 1u]; b = b - 1u; }
      read[b] = key;
    }
    for (var g = 0u; g < groups; g = g + 1u) {
      let lo = nRead * g / groups;
      let hi = nRead * (g + 1u) / groups;
      var gm = f32(0);
      for (var k = lo; k < hi; k = k + 1u) { gm = gm + response[read[k]]; }
      gm = gm / f32(hi - lo);
      for (var k = lo; k < hi; k = k + 1u) { level[read[k]] = estimate + reliability * (gm - mean) + offset; }
    }
  }
  let efficiency = f32(1) / (f32(1) + groupCost * f32(used - 1u));
  let teaching = max(f32(0), sessionMinutes - classMinutes - f32(0.5) * analysis);
  let anxiety = clamp(anxietyMean * (f32(1) - f32(0.5) * affective) + instAnxiety, f32(0), f32(1));
  let boost = (f32(1) + f32(0.25) * engagement) * (f32(1) + f32(0.2) * interests) * (f32(1) - f32(0.3) * anxiety);
  let plainBoost = f32(1) - f32(0.3) * clamp(anxietyMean, f32(0), f32(1));
  var gain = f32(0);
  var plainGain = f32(0);
  var proximal = f32(0);
  for (var i = 0u; i < n; i = i + 1u) {
    let task = atLogit(sharp * (theta[i] - level[i]), kA);
    let helped = shift(task, scaffold);
    gain = gain + teaching / sessionMinutes * efficiency * rate * chance(helped) * (f32(1) - chance(task)) * boost;
    let independent = inside(task, chanceCap(pHi));
    if (!independent && inside(helped, chanceCap(pLo))) { proximal = proximal + f32(1); }
    let plain = atLogit(sharp * (theta[i] - m0 - offset), kA);
    plainGain = plainGain + rate * chance(shift(plain, scaffold)) * (f32(1) - chance(plain)) * plainBoost;
  }
  stat(0u, gain / f32(n));
  stat(1u, plainGain / f32(n));
  stat(2u, proximal / f32(n));
  stat(6u, teaching);
)wgsl";

namespace syn::models {

namespace {
struct Prior {
  const char* name;
  double bias, sd;
};
const Prior kPriors[4] = {{"assumption (no information)", 0.8, 1.5}, {"research literature", 0.25, 0.7},
                          {"demographics and campus offices", 0.1, 0.6}, {"curriculum / course instructor", 0.0, 0.35}};
struct Instrument {
  const char* name;
  double classMinutes, readTime, noise, recall, social, participation, individual, affective, engagement, interests, anxiety;
};
const Instrument kInstruments[9] = {
    {"no pre-assessment", 0, 0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0},
    {"show of hands", 2, 0, 1.2, 0.4, 0.3, 0.9, 0, 0.2, 0.2, 0.0, 0.0},
    {"anonymous poll", 3, 0, 1.0, 0.4, 0.0, 0.8, 0, 0.4, 0.3, 0.1, 0.0},
    {"worksheet", 8, 0.25, 0.6, 0.0, 0.0, 0.95, 1, 0.1, 0.3, 0.1, 0.0},
    {"K-W-L chart", 7, 0.5, 0.9, 0.2, 0.1, 0.9, 1, 0.6, 0.6, 0.7, 0.0},
    {"concept map", 8, 0.6, 0.7, 0.0, 0.1, 0.9, 1, 0.2, 0.4, 0.4, 0.0},
    {"brainstorm", 5, 0.15, 1.1, 0.3, 0.2, 0.85, 0, 0.3, 0.5, 0.5, 0.0},
    {"formal quiz", 12, 0.1, 0.5, 0.3, 0.0, 0.95, 1, 0.0, 0.0, 0.0, 0.3},
    {"pre-assessment as homework", 0, 0, 0.6, 0.0, 0.0, 0.3, 1, 0.3, 0.3, 0.3, 0.0},
};
std::vector<double> row(const Instrument& i, const Prior& p, int size, double within, int groups) {
  return {i.classMinutes, i.readTime, i.noise, i.recall, i.social, i.participation, i.individual, i.affective, i.engagement, i.interests,
          i.anxiety, p.bias, p.sd, double(size), within, double(groups)};
}
const std::vector<double> kSizes = {6, 10, 15, 20, 30, 45, 60, 90, 120};
const std::vector<double> kSpreads = {0.5, 0.9, 1.4};
}  // namespace

const kernel::Model* audienceModel() {
  static const kernel::Model model = [] {
    kernel::Model m;
    m.name = "audience";
    m.title = "Identifying audience needs: priors, pre-assessment instruments and a one-shot session pitched at the ZPD";
    m.maxStats = 7;
    m.agentSource = audience_agent;
    m.compile = [](const Value& spec) {
      const Value& c = spec.sub("audience");
      const int size = std::max(1, std::min(120, c.integer("classSize", 25)));
      const double within = c.num("withinClassSd", 0.9);
      const int groups = std::max(1, c.integer("groups", 1));
      const std::vector<double> params = {c.num("communityMean", 0.0), c.num("betweenClassSd", 0.6), c.num("sessionMinutes", 50.0), c.num("analysisBudget", 3.0),
                                          c.num("scaffold", 1.5), c.num("sharpness", 2.0), c.num("groupCost", 0.05), c.num("anxietyMean", 0.4), 0.35, 0.8, 0.5};
      std::vector<std::vector<double>> table, bySize, grouping;
      for (const auto& i : kInstruments)
        for (const auto& p : kPriors) table.push_back(row(i, p, size, within, groups));
      for (int k : {2, 3, 4, 5})
        for (double s : kSizes) bySize.push_back(row(kInstruments[k], kPriors[1], static_cast<int>(s), within, 1));
      for (double sd : kSpreads)
        for (int g = 1; g <= 4; ++g) grouping.push_back(row(kInstruments[3], kPriors[1], size, sd, g));
      const std::uint32_t classes = static_cast<std::uint32_t>(c.integer("classes", 300));
      return std::vector<Plan>{makePlan("audience", classes, 7, params, table, seedOf(spec)), makePlan("audience", classes, 7, params, bySize, seedOf(spec)),
                               makePlan("audience", classes, 7, params, grouping, seedOf(spec))};
    };
    m.summarise = [](const Value&, const std::vector<Plan>& plans, const std::vector<Output>& out) {
      Value r = Value::object();
      std::vector<std::string> priors;
      for (const auto& p : kPriors) priors.push_back(p.name);
      r.set("priors", priors);
      const Plan& p = plans[0];
      const Output& o = out[0];
      Value table = Value::array();
      for (std::uint32_t i = 0; i < 9; ++i) {
        const auto& inst = kInstruments[i];
        std::vector<double> gain, value, rmse, zpd;
        double teaching = 0, analysed = 0;
        for (std::uint32_t k = 0; k < 4; ++k) {
          const std::uint32_t j = i * 4 + k;
          gain.push_back(o.mean(p, j, 0));
          value.push_back(o.mean(p, j, 0) - o.mean(p, j, 1));
          rmse.push_back(std::sqrt(o.mean(p, j, 3)));
          zpd.push_back(o.mean(p, j, 2));
          teaching = o.mean(p, j, 6);
          analysed = o.mean(p, j, 5);
        }
        table.push(Value::object()
                       .set("instrument", inst.name)
                       .set("classMinutes", inst.classMinutes)
                       .set("analysisPerLearner", inst.readTime)
                       .set("noise", inst.noise)
                       .set("bias", inst.recall + inst.social)
                       .set("participation", inst.participation)
                       .set("individual", inst.individual > 0.5)
                       .set("affective", inst.affective)
                       .set("engagement", inst.engagement)
                       .set("gain", Value::Numbers(gain))
                       .set("value", Value::Numbers(value))
                       .set("rmse", Value::Numbers(rmse))
                       .set("zpd", Value::Numbers(zpd))
                       .set("teachingMinutes", teaching)
                       .set("analysedShare", analysed));
      }
      r.set("instruments", table);
      Value bySize = Value::array();
      const Plan& q = plans[1];
      int idx = 0;
      for (int k : {2, 3, 4, 5}) {
        std::vector<double> read, rmse;
        for (std::size_t s = 0; s < kSizes.size(); ++s, ++idx) {
          read.push_back(out[1].mean(q, idx, 5));
          rmse.push_back(std::sqrt(out[1].mean(q, idx, 3)));
        }
        bySize.push(Value::object().set("instrument", kInstruments[k].name).set("read", Value::Numbers(read)).set("rmse", Value::Numbers(rmse)));
      }
      r.set("sizes", Value::Numbers(kSizes)).set("bySize", bySize);
      Value grouping = Value::array();
      const Plan& g = plans[2];
      for (std::size_t s = 0; s < kSpreads.size(); ++s) {
        std::vector<double> gains;
        for (std::uint32_t k = 0; k < 4; ++k) gains.push_back(out[2].mean(g, static_cast<std::uint32_t>(s * 4 + k), 0));
        grouping.push(Value::object().set("spread", kSpreads[s]).set("gain", Value::Numbers(gains)));
      }
      r.set("grouping", grouping);
      return r;
    };
    return m;
  }();
  return &model;
}

}  // namespace syn::models
