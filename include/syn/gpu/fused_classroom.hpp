#pragma once

// The fused classroom pipeline for GPUs (WebGPU / WGSL).
//
// One kernel invocation per simulated learner runs the whole course of the classroom engine:
// it samples the learner, applies needs, affect, motivation, the ZPD (in pure elliptic CGA:
// the task is a null vector, scaffolding a dilation versor, the zones caps), encoding and the
// spaced/retrieval review slots, the response to success and failure, and evaluates the test
// score on every reporting day. The 64 learners of a workgroup then reduce their statistics
// together; only per-workgroup partial sums leave the kernel (plus the per-learner scores the
// quantile kernel needs). Many designs run in one dispatch: the second grid dimension is the
// job (design, seed), so a whole Shapley allocation or sensitivity analysis is one batch.
//
// Everything the GPU needs is compiled here, on the CPU, into flat buffers (FusedPlan): a
// uniform header, the population/course/model constants, one row of levers per job, the job
// seeds and the reporting days. The host uploads them, dispatches three kernels
//   1. classroom-fused     (learner-workgroups × jobs)
//   2. reduce-partials     sums the workgroup partials into per-job means
//   3. score-quantiles     per (day, job) histogram of the scores → 10/50/90% quantiles
// and reads back two arrays (FusedOutput), which summarise() turns into ClassroomResults.
//
// Random numbers are counter-based (a PCG hash of seed, learner and draw index), so they are
// reproducible on any device and give common random numbers across jobs with the same seed.
// They differ from the CPU engine's xoshiro streams: GPU and WebAssembly results agree within
// Monte Carlo error, not bit for bit. runFusedReference() executes the same kernels on the CPU
// in 32-bit floats, draw for draw, so the kernel algorithm is tested natively without a GPU.

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "syn/classroom/classroom_engine.hpp"

namespace syn::gpu {

constexpr std::size_t kWorkgroupSize = 64;
constexpr std::size_t kMaxSessions = 16;    ///< units held in a learner's private memory
constexpr std::size_t kDayStats = 4;        ///< score, score², recall, at risk
constexpr std::size_t kSessionStats = 10;   ///< ability gain, effort, motivation, value, efficacy, anxiety, readiness, growth, engaged, in ZPD
constexpr std::size_t kJobStats = 5;        ///< end score, delayed score, ability gain, delayed score × high capital, high capital
constexpr std::size_t kParamCount = 40;
constexpr std::size_t kJobStride = 16;      ///< levers per job
constexpr std::size_t kHeaderWords = 12;
constexpr std::size_t kHistogramBins = 1024;
constexpr std::size_t kQuantileWorkgroupSize = 128;  ///< compatibility mode allows at most 128 invocations

struct FusedJob {
  InstructionalDesign design;
  std::uint32_t seed = 0;
};

struct FusedPlan {
  std::size_t numLearners = 0, numJobs = 0, sessions = 0;
  std::vector<double> days;  ///< reporting days (the delayed test is the last one)
  bool profiles = false;     ///< full daily grid and session statistics (else end and delayed test only)
  bool quantiles = false;    ///< run the score-quantile kernel
  std::size_t endDayIndex = 0, testDayIndex = 0;
  double lastSessionDay = 0.0, delayedTestDay = 0.0;

  // GPU buffers, in the exact binary layout of the WGSL kernels.
  std::vector<std::uint32_t> header;  ///< uniform block (kHeaderWords words)
  std::vector<float> params;          ///< population, course and model constants (kParamCount)
  std::vector<float> jobs;            ///< [job][kJobStride] lever values
  std::vector<std::uint32_t> seeds;   ///< [job]
  std::vector<float> dayTable;        ///< [day]

  std::size_t numDays() const { return days.size(); }
  std::size_t sessionOffset() const { return numDays() * kDayStats; }
  std::size_t jobOffset() const { return sessionOffset() + (profiles ? sessions * kSessionStats : 0); }
  /// Statistics per learner and per job.
  std::size_t statsPerJob() const { return jobOffset() + kJobStats; }
  std::size_t numWorkgroups() const { return (numLearners + kWorkgroupSize - 1) / kWorkgroupSize; }
  /// Workgroups to dispatch: fused (x, y = jobs), reduce (x), quantiles (x = days, y = jobs).
  std::size_t fusedDispatch() const { return numWorkgroups(); }
  std::size_t reduceDispatch() const { return (numJobs * statsPerJob() + 63) / 64; }
  std::size_t quantileDispatch() const { return quantiles ? numDays() : 0; }
  /// Buffer sizes in bytes.
  std::size_t scratchBytes() const { return numJobs * numLearners * statsPerJob() * 4; }
  std::size_t partialsBytes() const { return numJobs * numWorkgroups() * statsPerJob() * 4; }
  std::size_t sumsBytes() const { return numJobs * statsPerJob() * 4; }
  std::size_t quantileBytes() const { return std::max<std::size_t>(16, quantiles ? numJobs * numDays() * 3 * 4 : 16); }
};

/** What the GPU returns: per-job means of every statistic and, if requested, score quantiles. */
struct FusedOutput {
  std::vector<float> sums;       ///< [job][stat], already divided by the number of learners
  std::vector<float> quantiles;  ///< [job][day][q10, q50, q90]
};

/**
 * Compiles a batch of courses for the fused kernels. Throws std::invalid_argument when the
 * course exceeds the kernel's fixed limits (more than kMaxSessions sessions).
 */
FusedPlan compile(const PopulationSpec& population, const CourseSpec& course, std::size_t learners,
                  const std::vector<FusedJob>& jobs, bool profiles, bool quantiles);

/** Why `compile` cannot run this course on the GPU, or an empty string if it can. */
std::string limitation(const CourseSpec& course, std::size_t learners);

/**
 * The read-back of one job as a ClassroomResult: profiles, session traces and the summary.
 * Per-learner outcomes are not available (the kernels never return individual learners).
 */
ClassroomResult summarise(const FusedPlan& plan, const FusedOutput& output, std::size_t job);

/** Mean delayed test score of every job (the objective of allocation and sensitivities). */
std::vector<double> delayedScores(const FusedPlan& plan, const FusedOutput& output);

/** The fused kernel for a range of (job, workgroup) tiles on the CPU. */
struct FusedSlice {
  std::size_t tileBegin = 0, tileEnd = 0;  ///< tiles = job · numWorkgroups + workgroup
  std::vector<float> partials;             ///< [tile - tileBegin][stat]
  std::vector<float> scores;               ///< quantile plans: [tile - tileBegin][learner in tile][day]
};

/** Runs tiles [tileBegin, tileEnd) of the fused kernel on the CPU in single precision. */
FusedSlice runFusedTiles(const FusedPlan& plan, std::size_t tileBegin, std::size_t tileEnd);

/** The reduce and quantile kernels over slices covering every tile in order. */
FusedOutput finishFused(const FusedPlan& plan, const std::vector<FusedSlice>& slices);

/** The three kernels executed on the CPU in single precision, as the GPU runs them. */
FusedOutput runFusedReference(const FusedPlan& plan);

/** WGSL sources of the kernels. */
const std::string& classroomFusedKernel();
const std::string& reducePartialsKernel();
const std::string& scoreQuantilesKernel();

}  // namespace syn::gpu
