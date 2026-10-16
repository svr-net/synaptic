#pragma once

// Fused kernels: the one way every model of the library computes.
//
// A model is a population of agents (learners, classes, raters, grid points of a curve...) that
// each run the same program, the agent, in the space of syn/space/space.hpp. An agent reads the
// model's constants P(i), its job's row J(i) and counter-based random numbers U(k), and adds its
// statistics with stat(i, v); a workgroup of 64 agents sums its statistics into one partial, the
// partials are summed per job, and the model turns the per-job means into its result. Many jobs
// (designs, seeds, conditions, populations) run in one dispatch.
//
// The agent is written once (syn/kernel/agent.hpp) and runs on three engines:
//   WebGPU        the WGSL kernel on the GPU (32-bit floats);
//   CPU kernels   the same program compiled as C++ in 32-bit floats, tiles split across workers;
//   WebAssembly   the same program compiled as C++ in 64-bit floats.
// All three use the same random numbers (a PCG hash of seed, agent and draw index), so they agree
// to rounding, not just within Monte Carlo error.
//
// Kernel limits (WebGPU compatibility mode): 4 storage buffers per stage, 64 invocations per
// workgroup for the agents, 128 for the quantile kernel, at most kMaxStats statistics per agent.

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "syn/core/value.hpp"

namespace syn::kernel {

constexpr std::uint32_t kWorkgroup = 64;
constexpr std::uint32_t kQuantileWorkgroup = 128;
constexpr std::uint32_t kHistogramBins = 1024;
constexpr std::uint32_t kHeaderWords = 8;

/** One dispatch: agents × jobs of one model's agent program. */
struct Plan {
  std::string model;
  std::uint32_t agents = 0, jobs = 0;
  std::uint32_t stats = 0;      ///< statistics per agent (summed over agents)
  std::uint32_t quantiles = 0;  ///< values per agent kept for the 10/50/90% quantiles (in [0, 1])
  std::vector<float> params;    ///< model constants P(i)
  std::uint32_t jobStride = 0;  ///< floats per job row J(i)
  std::vector<float> jobRows;   ///< [job][jobStride]
  std::vector<std::uint32_t> seeds;  ///< [job]
  Value meta;                   ///< anything the model wants back in summarise()

  std::uint32_t workgroups() const { return (agents + kWorkgroup - 1) / kWorkgroup; }
  /** The uniform header of the kernels: agents, jobs, stats, quantiles, workgroups, params, stride, 0. */
  std::vector<std::uint32_t> header() const;
  /** params followed by the job rows: the constants buffer. */
  std::vector<float> constants() const;
  std::size_t scratchFloats() const { return static_cast<std::size_t>(jobs) * agents * quantiles; }
  std::size_t partialFloats() const { return static_cast<std::size_t>(jobs) * workgroups() * stats; }
  std::size_t sumFloats() const { return static_cast<std::size_t>(jobs) * stats; }
  std::size_t quantileFloats() const { return static_cast<std::size_t>(jobs) * quantiles * 3; }
};

/** What a dispatch returns: per-job means of every statistic, and quantiles if requested. */
struct Output {
  std::vector<double> means;      ///< [job][stat], divided by the number of agents
  std::vector<double> quantiles;  ///< [job][value][q10, q50, q90]

  double mean(const Plan& p, std::uint32_t job, std::uint32_t stat) const { return means[job * p.stats + stat]; }
  /** stats [first, first + n) of a job. */
  std::vector<double> range(const Plan& p, std::uint32_t job, std::uint32_t first, std::uint32_t n) const;
};

/** What an agent program sees (the C++ engines; the WGSL driver provides the same names). */
template <class T>
struct Ctx {
  const Plan* plan = nullptr;
  const float* params = nullptr;
  const float* row = nullptr;
  std::uint32_t seed = 0, job = 0, agent = 0;
  T* stats = nullptr;
  T* quants = nullptr;
};

using Agent32 = void (*)(Ctx<float>&);
using Agent64 = void (*)(Ctx<double>&);

/** A model: how to compile a specification, the agent program and how to read the result. */
struct Model {
  std::string name;
  std::string title;              ///< one line for the GPU page
  std::uint32_t maxStats = 0;     ///< size of the agent's private statistics array
  Agent32 agent32 = nullptr;
  Agent64 agent64 = nullptr;
  const char* agentSource = nullptr;  ///< the agent program as written (stringified)
  std::string helpersWgsl;        ///< model-specific WGSL helper functions (optional)
  std::function<std::vector<Plan>(const Value& spec)> compile;
  std::function<Value(const Value& spec, const std::vector<Plan>& plans, const std::vector<Output>& outputs)> summarise;
};

/** Every model of the library (src/models/all_models.cpp lists them). */
std::vector<const Model*> allModels();
/** All models, in the order of allModels(). */
const std::vector<const Model*>& models();
const Model& model(const std::string& name);

// ---------------------------------------------------------------- engines

enum class Precision { F32, F64 };

/** The partial sums of a range of (job, workgroup) tiles, as one worker computes them. */
struct Slice {
  std::size_t tileBegin = 0, tileEnd = 0;  ///< tile = job · workgroups + workgroup
  std::vector<double> partials;            ///< [tile − tileBegin][stat]
  std::vector<float> scratch;              ///< [tile − tileBegin][agent in tile][quantile value]
};

Slice runTiles(const Plan& plan, std::size_t tileBegin, std::size_t tileEnd, Precision precision);
/** The reduce and quantile kernels over slices that cover every tile in order. */
Output finish(const Plan& plan, const std::vector<Slice>& slices);
/** A whole plan on the CPU. */
Output runPlan(const Plan& plan, Precision precision);

/** Compile, run every plan on the CPU and summarise. */
Value run(const std::string& name, const Value& spec, Precision precision = Precision::F64);
/** The tiles of every plan of a model's specification that worker `part` of `parts` runs. */
Value runSlices(const std::string& name, const Value& spec, std::size_t part, std::size_t parts);
/** Summarises slices from runSlices (one array per worker, in worker order). */
Value analyseSlices(const std::string& name, const Value& spec, const std::vector<Value>& parts);
/** Summarises GPU read-backs: one { means, quantiles } per plan. */
Value analyseOutputs(const std::string& name, const Value& spec, const std::vector<Output>& outputs);

// ---------------------------------------------------------------- WGSL

/** The agent kernel of a model: space library, random numbers, the agent, the driver. */
std::string agentKernel(const Model& m);
/** Sums the workgroup partials of every (job, stat) and divides by the agents. */
const std::string& reduceKernel();
/** Per (value, job): histogram of the agents' values in [0, 1] → 10/50/90% quantiles. */
const std::string& quantileKernel();

}  // namespace syn::kernel
