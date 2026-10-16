#include "syn/kernel/kernel.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <memory>
#include <regex>
#include <stdexcept>

#include "syn/space/space.hpp"

namespace syn::kernel {

// ---------------------------------------------------------------- plans and outputs

std::vector<std::uint32_t> Plan::header() const {
  return {agents, jobs, stats, quantiles, workgroups(), static_cast<std::uint32_t>(params.size()), jobStride, 0u};
}

std::vector<float> Plan::constants() const {
  std::vector<float> c = params;
  c.insert(c.end(), jobRows.begin(), jobRows.end());
  if (c.empty()) c.push_back(0.0f);
  return c;
}

std::vector<double> Output::range(const Plan& p, std::uint32_t job, std::uint32_t first, std::uint32_t n) const {
  if (job >= p.jobs || first + n > p.stats) throw std::invalid_argument("Output::range: out of bounds");
  return std::vector<double>(means.begin() + job * p.stats + first, means.begin() + job * p.stats + first + n);
}

// ---------------------------------------------------------------- registry

namespace {
std::vector<const Model*> buildIndex() {
  std::vector<const Model*> v;
  for (const Model* m : allModels()) {
    for (const Model* x : v)
      if (x->name == m->name) throw std::logic_error("duplicate model " + m->name);
    v.push_back(m);
  }
  return v;
}
}  // namespace

const std::vector<const Model*>& models() {
  static const std::vector<const Model*> v = buildIndex();
  return v;
}

const Model& model(const std::string& name) {
  for (const auto* m : models())
    if (m->name == name) return *m;
  throw std::invalid_argument("unknown model: " + name);
}

// ---------------------------------------------------------------- CPU engines

namespace {

void validate(const Plan& p, const Model& m) {
  if (p.agents == 0 || p.jobs == 0) throw std::invalid_argument(m.name + ": a plan needs agents and jobs");
  if (p.stats == 0 || p.stats > m.maxStats) throw std::invalid_argument(m.name + ": statistics out of range");
  if (p.seeds.size() != p.jobs) throw std::invalid_argument(m.name + ": one seed per job");
  if (p.jobRows.size() != static_cast<std::size_t>(p.jobs) * p.jobStride) throw std::invalid_argument(m.name + ": job rows have the wrong size");
}

template <class T>
void runAgent(const Plan& p, const Model& m, std::uint32_t job, std::uint32_t a, T* stats, T* quants) {
  Ctx<T> c;
  c.plan = &p;
  c.params = p.params.data();
  c.row = p.jobRows.data() + static_cast<std::size_t>(job) * p.jobStride;
  c.seed = p.seeds[job];
  c.job = job;
  c.agent = a;
  c.stats = stats;
  c.quants = quants;
  if constexpr (std::is_same_v<T, float>) m.agent32(c);
  else m.agent64(c);
}

template <class T>
Slice tiles(const Plan& p, std::size_t begin, std::size_t end) {
  const Model& m = model(p.model);
  validate(p, m);
  const std::size_t total = static_cast<std::size_t>(p.jobs) * p.workgroups();
  if (begin > end || end > total) throw std::invalid_argument("runTiles: tile range out of bounds");
  Slice s;
  s.tileBegin = begin;
  s.tileEnd = end;
  s.partials.assign((end - begin) * p.stats, 0.0);
  s.scratch.assign((end - begin) * kWorkgroup * p.quantiles, 0.0f);
  std::vector<T> stats(m.maxStats), quants(std::max<std::uint32_t>(1, p.quantiles)), acc(p.stats);
  for (std::size_t tile = begin; tile < end; ++tile) {
    const auto job = static_cast<std::uint32_t>(tile / p.workgroups());
    const std::size_t wg = tile % p.workgroups();
    std::fill(acc.begin(), acc.end(), T(0));
    for (std::uint32_t k = 0; k < kWorkgroup; ++k) {
      const std::size_t a = wg * kWorkgroup + k;
      if (a >= p.agents) break;
      std::fill(stats.begin(), stats.end(), T(0));
      std::fill(quants.begin(), quants.end(), T(0));
      runAgent<T>(p, m, job, static_cast<std::uint32_t>(a), stats.data(), quants.data());
      for (std::uint32_t i = 0; i < p.stats; ++i) acc[i] = acc[i] + stats[i];
      for (std::uint32_t q = 0; q < p.quantiles; ++q)
        s.scratch[((tile - begin) * kWorkgroup + k) * p.quantiles + q] = static_cast<float>(quants[q]);
    }
    for (std::uint32_t i = 0; i < p.stats; ++i) s.partials[(tile - begin) * p.stats + i] = static_cast<double>(acc[i]);
  }
  return s;
}

}  // namespace

Slice runTiles(const Plan& p, std::size_t begin, std::size_t end, Precision precision) {
  return precision == Precision::F32 ? tiles<float>(p, begin, end) : tiles<double>(p, begin, end);
}

Output finish(const Plan& p, const std::vector<Slice>& slices) {
  const std::size_t total = static_cast<std::size_t>(p.jobs) * p.workgroups();
  std::size_t covered = 0;
  for (const auto& s : slices) {
    if (s.tileBegin != covered) throw std::invalid_argument("finish: slices must cover the tiles in order");
    if (s.partials.size() != (s.tileEnd - s.tileBegin) * p.stats) throw std::invalid_argument("finish: a slice has the wrong size");
    covered = s.tileEnd;
  }
  if (covered != total) throw std::invalid_argument("finish: slices do not cover every tile");
  auto slice = [&](std::size_t tile) -> const Slice& {
    for (const auto& s : slices)
      if (tile >= s.tileBegin && tile < s.tileEnd) return s;
    throw std::logic_error("finish: tile not covered");
  };
  Output out;
  out.means.assign(p.sumFloats(), 0.0);
  for (std::uint32_t job = 0; job < p.jobs; ++job)
    for (std::uint32_t i = 0; i < p.stats; ++i) {
      double acc = 0.0;
      for (std::uint32_t w = 0; w < p.workgroups(); ++w) {
        const std::size_t tile = static_cast<std::size_t>(job) * p.workgroups() + w;
        const Slice& s = slice(tile);
        acc += s.partials[(tile - s.tileBegin) * p.stats + i];
      }
      out.means[job * p.stats + i] = acc / p.agents;
    }
  if (p.keepAgents > 0 && p.quantiles > 0) {
    const std::uint32_t keep = std::min(p.keepAgents, p.agents);
    out.agentValues.assign(static_cast<std::size_t>(p.jobs) * keep * p.quantiles, 0.0);
    for (std::uint32_t job = 0; job < p.jobs; ++job)
      for (std::uint32_t a = 0; a < keep; ++a) {
        const std::size_t tile = static_cast<std::size_t>(job) * p.workgroups() + a / kWorkgroup;
        const Slice& s = slice(tile);
        for (std::uint32_t q = 0; q < p.quantiles; ++q)
          out.agentValues[(static_cast<std::size_t>(job) * keep + a) * p.quantiles + q] =
              s.scratch[((tile - s.tileBegin) * kWorkgroup + a % kWorkgroup) * p.quantiles + q];
      }
  }
  if (p.quantiles > 0) {
    out.quantiles.assign(p.quantileFloats(), 0.0);
    std::vector<std::uint32_t> hist(kHistogramBins);
    for (std::uint32_t job = 0; job < p.jobs; ++job)
      for (std::uint32_t q = 0; q < p.quantiles; ++q) {
        std::fill(hist.begin(), hist.end(), 0u);
        for (std::uint32_t a = 0; a < p.agents; ++a) {
          const std::size_t tile = static_cast<std::size_t>(job) * p.workgroups() + a / kWorkgroup;
          const Slice& s = slice(tile);
          const float v = std::min(1.0f, std::max(0.0f, s.scratch[((tile - s.tileBegin) * kWorkgroup + a % kWorkgroup) * p.quantiles + q]));
          ++hist[std::min<std::uint32_t>(kHistogramBins - 1, static_cast<std::uint32_t>(v * static_cast<float>(kHistogramBins)))];
        }
        const float qs[3] = {0.1f, 0.5f, 0.9f};
        for (int i = 0; i < 3; ++i) {
          const auto rank = static_cast<std::uint32_t>(std::floor(qs[i] * static_cast<float>(p.agents - 1)));
          std::uint32_t cum = 0, bin = 0;
          for (std::uint32_t b = 0; b < kHistogramBins; ++b) {
            cum += hist[b];
            if (cum > rank) {
              bin = b;
              break;
            }
          }
          out.quantiles[(static_cast<std::size_t>(job) * p.quantiles + q) * 3 + i] = (bin + 0.5) / kHistogramBins;
        }
      }
  }
  return out;
}

Output runPlan(const Plan& p, Precision precision) {
  return finish(p, {runTiles(p, 0, static_cast<std::size_t>(p.jobs) * p.workgroups(), precision)});
}

Value run(const std::string& name, const Value& spec, Precision precision) {
  const Model& m = model(name);
  const auto plans = m.compile(spec);
  std::vector<Output> outputs;
  for (const auto& p : plans) outputs.push_back(runPlan(p, precision));
  return m.summarise(spec, plans, outputs);
}

Value runSlices(const std::string& name, const Value& spec, std::size_t part, std::size_t parts) {
  if (parts == 0 || part >= parts) throw std::invalid_argument("runSlices: bad part");
  const Model& m = model(name);
  const auto plans = m.compile(spec);
  Value out = Value::array();
  for (const auto& p : plans) {
    const std::size_t total = static_cast<std::size_t>(p.jobs) * p.workgroups();
    const std::size_t begin = total * part / parts, end = total * (part + 1) / parts;
    const Slice s = runTiles(p, begin, end, Precision::F32);
    Value v = Value::object();
    v.set("tileBegin", static_cast<double>(s.tileBegin));
    v.set("tileEnd", static_cast<double>(s.tileEnd));
    v.set("partials", Value::Numbers(s.partials));
    v.set("scratch", Value::Numbers(s.scratch.begin(), s.scratch.end()));
    out.push(std::move(v));
  }
  return out;
}

Value analyseSlices(const std::string& name, const Value& spec, const std::vector<Value>& parts) {
  const Model& m = model(name);
  const auto plans = m.compile(spec);
  std::vector<Output> outputs;
  for (std::size_t i = 0; i < plans.size(); ++i) {
    std::vector<Slice> slices;
    for (const auto& part : parts) {
      const Value& v = part[i];
      Slice s;
      s.tileBegin = static_cast<std::size_t>(v.num("tileBegin", 0));
      s.tileEnd = static_cast<std::size_t>(v.num("tileEnd", 0));
      s.partials = v.nums("partials", {});
      const auto scratch = v.nums("scratch", {});
      s.scratch.assign(scratch.begin(), scratch.end());
      if (s.tileEnd > s.tileBegin) slices.push_back(std::move(s));
    }
    outputs.push_back(finish(plans[i], slices));
  }
  return m.summarise(spec, plans, outputs);
}

Value analyseOutputs(const std::string& name, const Value& spec, std::vector<Output> outputs, const std::vector<std::vector<double>>& scratch) {
  const Model& m = model(name);
  const auto plans = m.compile(spec);
  if (outputs.size() != plans.size()) throw std::invalid_argument(name + ": one GPU output per plan");
  for (std::size_t i = 0; i < plans.size(); ++i) {
    const Plan& p = plans[i];
    if (outputs[i].means.size() != p.sumFloats()) throw std::invalid_argument(name + ": GPU read-back has the wrong size");
    if (p.keepAgents > 0 && p.quantiles > 0) {
      if (i >= scratch.size() || scratch[i].size() < p.scratchFloats()) throw std::invalid_argument(name + ": the agent values were not read back");
      const std::uint32_t keep = std::min(p.keepAgents, p.agents);
      auto& v = outputs[i].agentValues;
      v.assign(static_cast<std::size_t>(p.jobs) * keep * p.quantiles, 0.0);
      for (std::uint32_t job = 0; job < p.jobs; ++job)
        for (std::uint32_t a = 0; a < keep; ++a)
          for (std::uint32_t q = 0; q < p.quantiles; ++q)
            v[(static_cast<std::size_t>(job) * keep + a) * p.quantiles + q] = scratch[i][(static_cast<std::size_t>(job) * p.agents + a) * p.quantiles + q];
    }
  }
  return m.summarise(spec, plans, outputs);
}

// ---------------------------------------------------------------- WGSL

namespace {

const char* kHeader = R"wgsl(
struct Header {
  agents : u32, jobs : u32, stats : u32, quant : u32,
  numWG : u32, nParams : u32, stride : u32, pad : u32,
};
)wgsl";

const char* kRandom = R"wgsl(
fn pcg(v : u32) -> u32 {
  let state = v * 747796405u + 2891336453u;
  let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
  return (word >> 22u) ^ word;
}
fn draw(s : u32, a : u32, k : u32) -> f32 {
  let h = pcg(s ^ pcg((a * 2654435769u) ^ pcg(k)));
  return (f32(h >> 8u) + 0.5) / 16777216.0;
}
)wgsl";

// The agent source is C++-compatible WGSL; translate the one construct that differs.
std::string toWgsl(const std::string& src) {
  static const std::regex arr(R"(ARR\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*,\s*([A-Za-z0-9_]+)\s*,\s*([0-9]+u?)\s*\))");
  std::string s = std::regex_replace(src, arr, "var $1 : array<$2, $3>");
  return s;
}

}  // namespace

std::string agentKernel(const Model& m) {
  std::string s = kHeader;
  s += R"wgsl(
@group(0) @binding(0) var<uniform> H : Header;
@group(0) @binding(1) var<storage, read> C : array<f32>;             // params | job rows
@group(0) @binding(2) var<storage, read> SEEDS : array<u32>;         // [job]
@group(0) @binding(3) var<storage, read_write> SCRATCH : array<f32>; // [job][agent][quantile value]
@group(0) @binding(4) var<storage, read_write> PARTIALS : array<f32>;// [job][workgroup][stat]
)wgsl";
  s += kRandom;
  s += space::spaceWgsl();
  s += "\nconst MAXS : u32 = " + std::to_string(m.maxStats) + "u;\n";
  s += R"wgsl(
var<private> S : array<f32, MAXS>;
var<private> A : u32;
var<private> JOB : u32;
var<private> NA : u32;
var<private> SEED : u32;
fn P(i : u32) -> f32 { return C[i]; }
fn J(i : u32) -> f32 { return C[H.nParams + JOB * H.stride + i]; }
fn U(k : u32) -> f32 { return draw(SEED, A, k); }
fn stat(i : u32, v : f32) { S[i] = S[i] + v; }
fn quant(q : u32, v : f32) { SCRATCH[(JOB * H.agents + A) * H.quant + q] = v; }
)wgsl";
  s += m.helpersWgsl;
  s += "\nfn agent() {\n" + toWgsl(m.agentSource) + "\n}\n";
  s += R"wgsl(
var<workgroup> RED : array<f32, 64>;
@compute @workgroup_size(64)
fn main(@builtin(local_invocation_id) lid : vec3<u32>, @builtin(workgroup_id) wid : vec3<u32>) {
  let a = wid.x * 64u + lid.x;
  A = a;
  JOB = wid.y;
  NA = H.agents;
  SEED = SEEDS[wid.y];
  for (var k = 0u; k < MAXS; k = k + 1u) { S[k] = 0.0; }
  if (a < H.agents) { agent(); }
  if (H.quant > 0x7fffffffu) { SCRATCH[0] = 0.0; }  // keeps the binding in the 'auto' layout of every model
  for (var k = 0u; k < H.stats; k = k + 1u) {
    RED[lid.x] = S[k];
    workgroupBarrier();
    for (var w = 32u; w > 0u; w = w >> 1u) {
      if (lid.x < w) { RED[lid.x] = RED[lid.x] + RED[lid.x + w]; }
      workgroupBarrier();
    }
    if (lid.x == 0u) { PARTIALS[(wid.y * H.numWG + wid.x) * H.stats + k] = RED[0]; }
    workgroupBarrier();
  }
}
)wgsl";
  return s;
}

const std::string& reduceKernel() {
  static const std::string s = std::string(kHeader) + R"wgsl(
@group(0) @binding(0) var<uniform> H : Header;
@group(0) @binding(1) var<storage, read> PARTIALS : array<f32>;
@group(0) @binding(2) var<storage, read_write> SUMS : array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x;
  if (i >= H.jobs * H.stats) { return; }
  let job = i / H.stats;
  let k = i % H.stats;
  var acc = 0.0;
  for (var w = 0u; w < H.numWG; w = w + 1u) { acc = acc + PARTIALS[(job * H.numWG + w) * H.stats + k]; }
  SUMS[i] = acc / f32(H.agents);
}
)wgsl";
  return s;
}

const std::string& quantileKernel() {
  static const std::string s = std::string(kHeader) + R"wgsl(
@group(0) @binding(0) var<uniform> H : Header;
@group(0) @binding(1) var<storage, read> SCRATCH : array<f32>;
@group(0) @binding(2) var<storage, read_write> OUT : array<f32>;
var<workgroup> HIST : array<atomic<u32>, 1024>;
@compute @workgroup_size(128)
fn main(@builtin(local_invocation_id) lid : vec3<u32>, @builtin(workgroup_id) wid : vec3<u32>) {
  let q = wid.x;
  let job = wid.y;
  for (var b = lid.x; b < 1024u; b = b + 128u) { atomicStore(&HIST[b], 0u); }
  workgroupBarrier();
  for (var a = lid.x; a < H.agents; a = a + 128u) {
    let v = clamp(SCRATCH[(job * H.agents + a) * H.quant + q], 0.0, 1.0);
    atomicAdd(&HIST[min(1023u, u32(v * 1024.0))], 1u);
  }
  workgroupBarrier();
  if (lid.x == 0u) {
    var qs = array<f32, 3>(0.1, 0.5, 0.9);
    for (var i = 0u; i < 3u; i = i + 1u) {
      let rank = u32(floor(qs[i] * f32(H.agents - 1u)));
      var cum = 0u;
      var bin = 0u;
      for (var b = 0u; b < 1024u; b = b + 1u) {
        cum = cum + atomicLoad(&HIST[b]);
        if (cum > rank) { bin = b; break; }
      }
      OUT[(job * H.quant + q) * 3u + i] = (f32(bin) + 0.5) / 1024.0;
    }
  }
}
)wgsl";
  return s;
}

}  // namespace syn::kernel
