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
// The agent is written once, in WGSL, and there is no other implementation of it. Three WGSL
// kernels (agent, reduce, quantile; see agentKernel) run on a GPU through WebGPU, or — where there
// is none, and in the native build and its tests — on the software GPU (syn/gpu/software_gpu.hpp),
// which executes the same WGSL as a GPU does.
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
  std::uint32_t keepAgents = 0; ///< agents (from 0) whose kept values are also returned one by one
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
  std::vector<double> agentValues;  ///< [job][kept agent][value] when the plan keeps agents

  double mean(const Plan& p, std::uint32_t job, std::uint32_t stat) const { return means[job * p.stats + stat]; }
  /** stats [first, first + n) of a job. */
  std::vector<double> range(const Plan& p, std::uint32_t job, std::uint32_t first, std::uint32_t n) const;
};

/** A model: how to compile a specification, the agent program (WGSL) and how to read the result. */
struct Model {
  std::string name;
  std::string title;              ///< one line for the GPU page
  std::uint32_t maxStats = 0;     ///< size of the agent's private statistics array
  const char* agentSource = nullptr;  ///< the agent program: the body of fn agent() in WGSL
  std::string helpersWgsl;        ///< model-specific WGSL helper functions (optional)
  std::function<std::vector<Plan>(const Value& spec)> compile;
  std::function<Value(const Value& spec, const std::vector<Plan>& plans, const std::vector<Output>& outputs)> summarise;
};

/** Every model of the library (src/models/all_models.cpp lists them). */
std::vector<const Model*> allModels();
/** All models, in the order of allModels(). */
const std::vector<const Model*>& models();
const Model& model(const std::string& name);

// ---------------------------------------------------------------- the software GPU

/** One plan on the software GPU: the agent, reduce and quantile kernels, as WebGPU runs them. */
Output simulate(const Plan& plan);
/** Compile a model's specification, run every plan on the software GPU and summarise. */
Value run(const std::string& name, const Value& spec);
/**
 * Worker `part` of `parts`: the agent kernel over its share of the workgroups of every plan, on
 * the software GPU. Returns, per plan, the words it wrote: { scratch, partials } as runs
 * [offset, values...].
 */
Value simulateSlices(const std::string& name, const Value& spec, std::size_t part, std::size_t parts);
/** Merges the workers' slices (in any order), runs the reduce and quantile kernels and summarises. */
Value analyseSlices(const std::string& name, const Value& spec, const std::vector<Value>& parts);
/**
 * Summarises WebGPU read-backs: one Output per plan (means and quantiles); when a plan keeps
 * agents, `scratch` holds its whole agent-values buffer, [job][agent][value].
 */
Value analyseOutputs(const std::string& name, const Value& spec, std::vector<Output> outputs,
                     const std::vector<std::vector<double>>& scratch = {});

// ---------------------------------------------------------------- WGSL

/**
 * The agent kernel of a model: the space, the random numbers, the agent interface, the agent and
 * the driver that reduces each workgroup's statistics. Interface of the agent:
 *   A, JOB, NA, SEED  the agent, its job, the number of agents, the job's seed (u32)
 *   P(i), J(i)        a model constant, a value of the job's row (f32)
 *   U(k)              the k-th uniform in (0, 1) of this agent under the job's seed
 *   jobDraw(s, i, k)  a uniform shared by every agent of the job (fixed sets: test items, raters)
 *   stat(i, v)        adds v to statistic i;   quant(q, v) keeps value q (in [0, 1]) for quantiles
 */
std::string agentKernel(const Model& m);
/** Sums the workgroup partials of every (job, stat) and divides by the agents. */
const std::string& reduceKernel();
/** Per (value, job): histogram of the agents' values in [0, 1] → 10/50/90% quantiles. */
const std::string& quantileKernel();

}  // namespace syn::kernel
