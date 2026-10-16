#include <cmath>

#include "syn/classroom/allocation.hpp"
#include "syn/classroom/classroom_engine.hpp"
#include "syn/classroom/sensitivities.hpp"
#include "syn/gpu/fused_classroom.hpp"
#include "test_framework.hpp"

using namespace syn;

namespace {
const CourseSpec kCourse{};
}  // namespace

TEST(gpu_plan_layout) {
  const auto plan = gpu::compile(PopulationSpec::secondary(), kCourse, 130, {{theoryDesign(Theory::Blended), 7u}}, true, true);
  CHECK(plan.header.size() == gpu::kHeaderWords);
  CHECK(plan.params.size() == gpu::kParamCount);
  CHECK(plan.jobs.size() == gpu::kJobStride);
  CHECK(plan.numWorkgroups() == 3);
  CHECK_NEAR(plan.days[plan.testDayIndex], 79.0, 1e-12);
  CHECK_NEAR(plan.days[plan.endDayIndex], 50.0, 1e-12);
  CHECK(plan.statsPerJob() == plan.numDays() * 4 + 8 * 10 + 5);
  CHECK(plan.header[4] == plan.statsPerJob());
  const auto small = gpu::compile(PopulationSpec::secondary(), kCourse, 64, {{theoryDesign(Theory::Banking), 1u}}, false, false);
  CHECK(small.numDays() == 2 && small.statsPerJob() == 2 * 4 + 5);
  CourseSpec long_ = kCourse;
  long_.sessions = 20;
  CHECK(!gpu::limitation(long_, 100).empty());
  CHECK_THROWS(gpu::compile(PopulationSpec::secondary(), long_, 100, {{theoryDesign(Theory::Banking), 1u}}, false, false));
  for (const auto* k : {&gpu::classroomFusedKernel(), &gpu::reducePartialsKernel(), &gpu::scoreQuantilesKernel()})
    CHECK(k->find("@compute") != std::string::npos);
}

TEST(gpu_reference_agrees_with_the_engine) {
  // Same model, different random numbers (counter-based hash vs xoshiro) and f32 vs f64:
  // agreement within Monte Carlo error.
  const int n = 2000;
  for (Theory t : {Theory::Banking, Theory::Cognitivism, Theory::Blended}) {
    for (const auto& pop : {PopulationSpec::children(), PopulationSpec::adults()}) {
      const ClassroomEngine engine(pop, kCourse, SimulationSettings{n, 2024});
      const ClassroomResult cpu = engine.run(theoryDesign(t));
      const auto plan = gpu::compile(pop, kCourse, n, {{theoryDesign(t), 2024u}}, true, true);
      const ClassroomResult g = gpu::summarise(plan, gpu::runFusedReference(plan), 0);
      const double tol = 4.0 * std::max(0.01, cpu.score.q90.back() - cpu.score.q10.back()) / 2.56 / std::sqrt(n) + 0.004;
      CHECK_NEAR(g.delayedScore, cpu.delayedScore, tol);
      CHECK_NEAR(g.endScore, cpu.endScore, 1.5 * tol);
      CHECK_NEAR(g.abilityGain, cpu.abilityGain, 0.05 + 0.03 * std::fabs(cpu.abilityGain));
      CHECK_NEAR(g.score.q50.back(), cpu.score.q50.back(), 2 * tol + 0.002);
      CHECK_NEAR(g.sessions.effort.back(), cpu.sessions.effort.back(), 0.02);
      CHECK_NEAR(g.sessions.zpdShare.back(), cpu.sessions.zpdShare.back(), 0.04);
      CHECK_NEAR(g.equityGap, cpu.equityGap, 0.03);
      for (std::size_t k = 0; k < g.score.days.size(); ++k) CHECK(g.score.q10[k] <= g.score.q50[k] && g.score.q50[k] <= g.score.q90[k]);
    }
  }
}

TEST(gpu_jobs_batch_and_common_random_numbers) {
  // A batch of jobs equals the jobs run one by one, and jobs with the same seed share learners.
  const std::vector<gpu::FusedJob> jobs = {{theoryDesign(Theory::Banking), 5u}, {theoryDesign(Theory::Blended), 5u}, {theoryDesign(Theory::Blended), 9u}};
  const auto batch = gpu::compile(PopulationSpec::secondary(), kCourse, 300, jobs, false, false);
  const auto all = gpu::delayedScores(batch, gpu::runFusedReference(batch));
  for (std::size_t j = 0; j < jobs.size(); ++j) {
    const auto one = gpu::compile(PopulationSpec::secondary(), kCourse, 300, {jobs[j]}, false, false);
    CHECK(gpu::delayedScores(one, gpu::runFusedReference(one))[0] == all[j]);
  }
  CHECK(all[1] > all[0] + 0.2);
  CHECK(all[2] != all[1]);
  // Splitting the tiles across workers does not change the result.
  const std::size_t tiles = batch.numJobs * batch.numWorkgroups();
  const auto split = gpu::finishFused(batch, {gpu::runFusedTiles(batch, 0, 4), gpu::runFusedTiles(batch, 4, 7), gpu::runFusedTiles(batch, 7, tiles)});
  CHECK(gpu::delayedScores(batch, split) == all);
  CHECK_THROWS(gpu::finishFused(batch, {gpu::runFusedTiles(batch, 0, 4)}));
}

TEST(gpu_allocation_and_sensitivities_from_a_batch) {
  const auto pop = PopulationSpec::secondary();
  const auto q = allocationRequest(theoryDesign(Theory::Blended), theoryDesign(Theory::Banking), 6);
  std::vector<gpu::FusedJob> jobs;
  for (const auto& d : q.designs) jobs.push_back({d, 2024u});
  const auto plan = gpu::compile(pop, kCourse, 200, jobs, false, false);
  const auto r = allocateFromScores(q, gpu::delayedScores(plan, gpu::runFusedReference(plan)));
  CHECK_NEAR(r.shapleySum, r.totalGain, 1e-12);
  CHECK(r.totalGain > 0.2);
  const auto s = sensitivityRequest(theoryDesign(Theory::Blended), 0.1, 3, 2024);
  std::vector<gpu::FusedJob> runs;
  for (const auto& run : s.runs) runs.push_back({run.design, static_cast<std::uint32_t>(run.seed)});
  const auto sp = gpu::compile(pop, kCourse, 150, runs, false, false);
  const auto sens = sensitivitiesFromScores(s, gpu::delayedScores(sp, gpu::runFusedReference(sp)));
  CHECK(sens.meanVarianceReduction > 5.0);  // common random numbers on the GPU too
}
