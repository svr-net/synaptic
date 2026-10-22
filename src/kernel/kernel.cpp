#include "syn/kernel/kernel.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <memory>
#include <stdexcept>
#include <thread>

#include "../core/compute_parallel.hpp"

#include "syn/gpu/software_gpu.hpp"
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

// ---------------------------------------------------------------- the software GPU

namespace {

void validate(const Plan& p, const Model& m) {
  if (p.agents == 0 || p.jobs == 0) throw std::invalid_argument(m.name + ": a plan needs agents and jobs");
  if (p.stats == 0 || p.stats > m.maxStats) throw std::invalid_argument(m.name + ": statistics out of range");
  if (p.seeds.size() != p.jobs) throw std::invalid_argument(m.name + ": one seed per job");
  if (p.jobRows.size() != static_cast<std::size_t>(p.jobs) * p.jobStride) throw std::invalid_argument(m.name + ": job rows have the wrong size");
}

// Each model's agent kernel, and the reduce and quantile kernels, compiled once.
struct Shaders {
  std::shared_ptr<const gpu::Shader> agent, reduce, quantile;
};

Shaders shaders(const Model& m) {   // no lock: a model's kernel compiled by whichever thread asks first (compute::Lazy)
  static compute::Lazy<std::string, std::shared_ptr<const gpu::Shader>> agents(256);
  static const auto reduce = gpu::Shader::compile(reduceKernel());
  static const auto quantile = gpu::Shader::compile(quantileKernel());
  return {agents.get(m.name, [&] { return gpu::Shader::compile(agentKernel(m)); }), reduce, quantile};
}

// The buffers of a plan, as the WebGPU host creates them (at least 16 bytes each).
struct Buffers {
  gpu::Buffer header, constants, seeds, scratch, partials;
};

gpu::Buffer atLeast4(gpu::Buffer b) {
  if (b.words.size() < 4) b.words.resize(4, 0u);
  return b;
}

Buffers buffers(const Plan& p, bool trackWrites) {
  Buffers b;
  b.header = atLeast4(gpu::Buffer::ofWords(p.header()));
  b.constants = atLeast4(gpu::Buffer::ofFloats(p.constants()));
  b.seeds = atLeast4(gpu::Buffer::ofWords(p.seeds));
  b.scratch = gpu::Buffer::zeros(std::max<std::size_t>(4, p.scratchFloats()), trackWrites);
  b.partials = gpu::Buffer::zeros(std::max<std::size_t>(4, p.partialFloats()), trackWrites);
  return b;
}

// The agent kernel over workgroups [first, last). Natively the workgroups are spread over the
// cores (they write disjoint words, as on a GPU) with compute's lanes -- no lock, no blocking wait;
// in WebAssembly the page spreads them over workers instead (simulateSlices).
void agents(const Plan& p, const Shaders& sh, Buffers& b, std::size_t first, std::size_t last) {
  const std::vector<gpu::Buffer*> bind = {&b.header, &b.constants, &b.seeds, &b.scratch, &b.partials};
#if defined(__EMSCRIPTEN__)
  gpu::dispatch(*sh.agent, bind, p.workgroups(), p.jobs, 1, first, last);
#else
  const std::size_t groups = last - first;
  const std::size_t threads = std::min<std::size_t>(std::max(1u, std::thread::hardware_concurrency()), groups / 4 + 1);
  if (threads <= 1) {
    gpu::dispatch(*sh.agent, bind, p.workgroups(), p.jobs, 1, first, last);
    return;
  }
  compute::lanes(threads, [&](std::size_t t) {
    gpu::dispatch(*sh.agent, bind, p.workgroups(), p.jobs, 1, first + groups * t / threads, first + groups * (t + 1) / threads);
  }, threads);
#endif
}

// The reduce and quantile kernels, then the read-back.
Output finishPlan(const Plan& p, const Shaders& sh, Buffers& b) {
  gpu::Buffer sums = gpu::Buffer::zeros(std::max<std::size_t>(4, p.sumFloats()));
  gpu::Buffer quantiles = gpu::Buffer::zeros(std::max<std::size_t>(4, p.quantileFloats()));
  gpu::dispatch(*sh.reduce, {&b.header, &b.partials, &sums}, static_cast<std::uint32_t>((p.sumFloats() + 63) / 64), 1, 1);
  if (p.quantiles > 0) gpu::dispatch(*sh.quantile, {&b.header, &b.scratch, &quantiles}, p.quantiles, p.jobs, 1);
  Output out;
  const auto s = sums.floats(), q = quantiles.floats();
  out.means.assign(s.begin(), s.begin() + static_cast<std::ptrdiff_t>(p.sumFloats()));
  if (p.quantiles > 0) out.quantiles.assign(q.begin(), q.begin() + static_cast<std::ptrdiff_t>(p.quantileFloats()));
  if (p.keepAgents > 0 && p.quantiles > 0) {
    const auto scratch = b.scratch.floats();
    const std::uint32_t keep = std::min(p.keepAgents, p.agents);
    out.agentValues.assign(static_cast<std::size_t>(p.jobs) * keep * p.quantiles, 0.0);
    for (std::uint32_t job = 0; job < p.jobs; ++job)
      for (std::uint32_t a = 0; a < keep; ++a)
        for (std::uint32_t v = 0; v < p.quantiles; ++v)
          out.agentValues[(static_cast<std::size_t>(job) * keep + a) * p.quantiles + v] =
              scratch[(static_cast<std::size_t>(job) * p.agents + a) * p.quantiles + v];
  }
  return out;
}

// The words a dispatch wrote, as runs [offset, values...].
Value writtenRuns(const gpu::Buffer& b) {
  Value runs = Value::array();
  const std::size_t n = b.words.size();
  std::size_t i = 0;
  while (i < n) {
    if (!b.written[i]) {
      ++i;
      continue;
    }
    std::vector<double> run = {static_cast<double>(i)};
    for (; i < n && b.written[i]; ++i) run.push_back(static_cast<double>(gpu::asFloat(b.words[i])));
    runs.push(Value::Numbers(std::move(run)));
  }
  return runs;
}

void applyRuns(gpu::Buffer& b, const Value& runs) {
  if (!runs.isArray()) return;
  for (const auto& r : runs.asArray()) {
    const auto& v = r.asNumbers();
    if (v.empty()) continue;
    const auto at = static_cast<std::size_t>(v[0]);
    if (at + v.size() - 1 > b.words.size()) throw std::invalid_argument("analyseSlices: a slice does not fit its buffer");
    for (std::size_t k = 1; k < v.size(); ++k) b.words[at + k - 1] = gpu::asWord(static_cast<float>(v[k]));
  }
}

}  // namespace

Output simulate(const Plan& p) {
  const Model& m = model(p.model);
  validate(p, m);
  const Shaders sh = shaders(m);
  Buffers b = buffers(p, false);
  agents(p, sh, b, 0, static_cast<std::size_t>(p.jobs) * p.workgroups());
  return finishPlan(p, sh, b);
}

Value run(const std::string& name, const Value& spec) {
  const Model& m = model(name);
  const auto plans = m.compile(spec);
  std::vector<Output> outputs;
  for (const auto& p : plans) outputs.push_back(simulate(p));
  return m.summarise(spec, plans, outputs);
}

Value simulateSlices(const std::string& name, const Value& spec, std::size_t part, std::size_t parts) {
  if (parts == 0 || part >= parts) throw std::invalid_argument("simulateSlices: bad part");
  const Model& m = model(name);
  const auto plans = m.compile(spec);
  Value out = Value::array();
  for (const auto& p : plans) {
    validate(p, m);
    const std::size_t total = static_cast<std::size_t>(p.jobs) * p.workgroups();
    Buffers b = buffers(p, true);
    agents(p, shaders(m), b, total * part / parts, total * (part + 1) / parts);
    Value v = Value::object();
    v.set("scratch", writtenRuns(b.scratch));
    v.set("partials", writtenRuns(b.partials));
    out.push(std::move(v));
  }
  return out;
}

Value analyseSlices(const std::string& name, const Value& spec, const std::vector<Value>& parts) {
  const Model& m = model(name);
  const auto plans = m.compile(spec);
  std::vector<Output> outputs;
  for (std::size_t i = 0; i < plans.size(); ++i) {
    const Plan& p = plans[i];
    validate(p, m);
    Buffers b = buffers(p, false);
    for (const auto& part : parts) {
      if (i >= part.size()) throw std::invalid_argument("analyseSlices: a worker returned too few plans");
      const Value slice = part[i];
      applyRuns(b.scratch, slice.at("scratch"));
      applyRuns(b.partials, slice.at("partials"));
    }
    outputs.push_back(finishPlan(p, shaders(m), b));
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
// A uniform in (0, 1): 23 bits, so that it stays strictly inside the interval in f32
// (with 24 bits, (2^24 − 1 + ½) rounds to 2^24 and the draw would be exactly 1).
fn draw(s : u32, a : u32, k : u32) -> f32 {
  let h = pcg(s ^ pcg((a * 2654435769u) ^ pcg(k)));
  return (f32(h >> 9u) + 0.5) / 8388608.0;
}
)wgsl";

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
var<private> S : array<f32, MAXS>;    // zero at the start of every invocation (WGSL private memory)
var<private> A : u32;
var<private> JOB : u32;
var<private> NA : u32;
var<private> SEED : u32;
fn P(i : u32) -> f32 { return C[i]; }
fn J(i : u32) -> f32 { return C[H.nParams + JOB * H.stride + i]; }
fn U(k : u32) -> f32 { return draw(SEED, A, k); }
fn jobDraw(s : u32, i : u32, k : u32) -> f32 { return draw(SEED ^ s, i, k); }
fn stat(i : u32, v : f32) { S[i] = S[i] + v; }
fn quant(q : u32, v : f32) { SCRATCH[(JOB * H.agents + A) * H.quant + q] = v; }
)wgsl";
  s += m.helpersWgsl;
  s += "\nfn agent() {\n" + std::string(m.agentSource) + "\n}\n";
  s += R"wgsl(
// The workgroup's statistics are a matrix [64 agents × stats]. It is reduced in tiles of
// 32 statistics: every agent writes its row of the tile to workgroup memory (rows padded to 33
// words, so that neighbouring agents write different banks), then 32 agents each sum a column.
// Two barriers per tile instead of seven per statistic.
const TILE : u32 = 32u;
const PITCH : u32 = 33u;
var<workgroup> TILE_SUMS : array<f32, 2112>;  // 64 × PITCH
@compute @workgroup_size(64)
fn main(@builtin(local_invocation_id) lid : vec3<u32>, @builtin(workgroup_id) wid : vec3<u32>) {
  let a = wid.x * 64u + lid.x;
  A = a;
  JOB = wid.y;
  NA = H.agents;
  SEED = SEEDS[wid.y];
  if (a < H.agents) { agent(); }
  if (H.quant > 0x7fffffffu) { SCRATCH[0] = 0.0; }  // keeps the binding in the 'auto' layout of every model
  let row = lid.x * PITCH;
  let first = (wid.y * H.numWG + wid.x) * H.stats;
  for (var k0 = 0u; k0 < H.stats; k0 = k0 + TILE) {
    for (var j = 0u; j < TILE; j = j + 1u) { TILE_SUMS[row + j] = S[min(k0 + j, MAXS - 1u)]; }
    workgroupBarrier();
    if (lid.x < TILE && k0 + lid.x < H.stats) {
      var acc = 0.0;
      for (var r = 0u; r < 64u; r = r + 1u) { acc = acc + TILE_SUMS[r * PITCH + lid.x]; }
      PARTIALS[first + k0 + lid.x] = acc;
    }
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
