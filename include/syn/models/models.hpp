#pragma once

// Shared helpers for the models: building plans from a specification and reading results.
// Every model lives in src/models/<name>.cpp: its agent (SYN_AGENT), how a specification
// compiles to plans, and how the per-job means become the result.

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "syn/core/value.hpp"
#include "syn/kernel/kernel.hpp"

namespace syn::models {

using kernel::Output;
using kernel::Plan;

/** The seed of a specification (sim.seed, default 2024). */
inline std::uint32_t seedOf(const Value& spec) {
  return static_cast<std::uint32_t>(spec.sub("sim").num("seed", 2024.0));
}

/** sim.learners (default `fallback`), at least 2. */
inline std::uint32_t learnersOf(const Value& spec, int fallback) {
  const int n = spec.sub("sim").integer("learners", fallback);
  return static_cast<std::uint32_t>(n < 2 ? 2 : n);
}

/** A plan with one job row per entry of `rows` (all of the same length) and one seed for all. */
inline Plan makePlan(const std::string& model, std::uint32_t agents, std::uint32_t stats, std::vector<double> params,
                     const std::vector<std::vector<double>>& rows, std::uint32_t seed, std::uint32_t quantiles = 0) {
  Plan p;
  p.model = model;
  p.agents = agents;
  p.stats = stats;
  p.quantiles = quantiles;
  for (double x : params) p.params.push_back(static_cast<float>(x));
  if (p.params.empty()) p.params.push_back(0.0f);
  p.jobs = static_cast<std::uint32_t>(rows.size());
  p.jobStride = rows.empty() ? 0 : static_cast<std::uint32_t>(rows[0].size());
  for (const auto& r : rows) {
    if (r.size() != p.jobStride) throw std::invalid_argument(model + ": job rows of different lengths");
    for (double x : r) p.jobRows.push_back(static_cast<float>(x));
  }
  p.seeds.assign(p.jobs, seed);
  return p;
}

/** Evenly spaced values. */
inline std::vector<double> grid(double a, double b, int n) {
  std::vector<double> v;
  for (int i = 0; i < n; ++i) v.push_back(n == 1 ? a : a + (b - a) * i / (n - 1));
  return v;
}

/** Statistic `stat` of every job of a plan. */
inline std::vector<double> column(const Plan& p, const Output& o, std::uint32_t stat) {
  std::vector<double> v;
  for (std::uint32_t j = 0; j < p.jobs; ++j) v.push_back(o.mean(p, j, stat));
  return v;
}

}  // namespace syn::models

/** Fills the agent fields of a kernel::Model from an agent defined with SYN_AGENT. */
#define SYN_USE_AGENT(MODEL, AGENT)                     \
  (MODEL).agent32 = &::syn_agents::AGENT<float>;        \
  (MODEL).agent64 = &::syn_agents::AGENT<double>;       \
  (MODEL).agentSource = ::syn_agents::AGENT##_source()
