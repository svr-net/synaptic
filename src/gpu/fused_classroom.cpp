#include "syn/gpu/fused_classroom.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "syn/humanism/maslow.hpp"
#include "syn/social/zpd.hpp"

namespace syn::gpu {

std::string limitation(const CourseSpec& course, std::size_t learners) {
  if (course.sessions < 1) return "the course needs at least one session";
  if (static_cast<std::size_t>(course.sessions) > kMaxSessions) return "more than 16 sessions (the kernel keeps 16 units per learner)";
  if (learners < 2) return "at least two learners are needed";
  return {};
}

FusedPlan compile(const PopulationSpec& population, const CourseSpec& course, std::size_t learners,
                  const std::vector<FusedJob>& jobs, bool profiles, bool quantiles) {
  const std::string why = limitation(course, learners);
  if (!why.empty()) throw std::invalid_argument("gpu::compile: " + why);
  if (jobs.empty()) throw std::invalid_argument("gpu::compile: no jobs");
  if (course.sessionGap <= 0.0 || course.delayedTest < 0.0 || course.reportStep <= 0.0)
    throw std::invalid_argument("gpu::compile: need sessionGap > 0, delayedTest >= 0, reportStep > 0");

  FusedPlan p;
  p.numLearners = learners;
  p.numJobs = jobs.size();
  p.sessions = static_cast<std::size_t>(course.sessions);
  p.profiles = profiles;
  p.quantiles = quantiles && profiles;
  p.lastSessionDay = course.sessionGap * (course.sessions - 1);
  p.delayedTestDay = p.lastSessionDay + course.delayedTest;
  const double endDay = p.lastSessionDay + 1.0;
  // The same reporting days as the classroom engine.
  if (profiles) {
    for (double t = 0.0; t <= p.delayedTestDay + 1e-9; t += course.reportStep) p.days.push_back(t);
    p.days.push_back(endDay);
    p.days.push_back(p.delayedTestDay);
    std::sort(p.days.begin(), p.days.end());
    p.days.erase(std::unique(p.days.begin(), p.days.end(), [](double a, double b) { return std::fabs(a - b) < 1e-9; }), p.days.end());
  } else {
    p.days = {endDay, p.delayedTestDay};
  }
  for (std::size_t k = 0; k < p.days.size(); ++k) {
    if (std::fabs(p.days[k] - endDay) < 1e-9) p.endDayIndex = k;
    if (std::fabs(p.days[k] - p.delayedTestDay) < 1e-9) p.testDayIndex = k;
  }

  const MemoryParams m;
  const MaslowParams maslow;
  const ZpdParams zpd;
  p.params.assign(kParamCount, 0.0f);
  const double values[] = {population.age, population.ageSd, population.priorMean, population.priorSd, population.anxietyMean,
                           population.anxietySd, population.growthMean, population.growthSd, population.efficacyMean,
                           population.efficacySd, population.intrinsicMean, population.intrinsicSd, population.capitalMean,
                           population.capitalSd, population.capitalPrior,
                           course.sessionGap, course.difficulty, course.difficultyStep, course.abstraction, course.atRiskBelow,
                           course.difficulty + course.difficultyStep * (course.sessions - 1),
                           m.initialStability, m.restudyFloor, m.restudyGain, m.retrievalFloor, m.retrievalGain, m.failureFactor,
                           m.minStability, m.pretestBoost,
                           maslow.threshold, maslow.width, zpd.independentThreshold, zpd.assistedThreshold,
                           std::min(1.0, std::max(0.0, population.capitalMean)),  // capital split: the median of the clamped normal
                           0.9, 12.0};                                            // Piaget: onset sd, formal-operations onset
  for (std::size_t i = 0; i < sizeof(values) / sizeof(values[0]); ++i) p.params[i] = static_cast<float>(values[i]);

  for (const auto& j : jobs) {
    const InstructionalDesign d = j.design.clamped();
    for (int k = 0; k < kLevers; ++k) p.jobs.push_back(static_cast<float>(d.lever[k]));
    p.seeds.push_back(j.seed);
  }
  for (double d : p.days) p.dayTable.push_back(static_cast<float>(d));

  const auto u32 = [](std::size_t v) { return static_cast<std::uint32_t>(v); };
  p.header = {u32(p.numLearners), u32(p.numJobs), u32(p.sessions), u32(p.numDays()),
              u32(p.statsPerJob()), p.profiles ? 1u : 0u, u32(p.numWorkgroups()), u32(p.sessionOffset()),
              u32(p.jobOffset()), u32(p.endDayIndex), u32(p.testDayIndex), u32(kParamCount + p.numDays())};
  return p;
}

ClassroomResult summarise(const FusedPlan& p, const FusedOutput& out, std::size_t job) {
  if (job >= p.numJobs) throw std::invalid_argument("gpu::summarise: no such job");
  const std::size_t stride = p.statsPerJob();
  if (out.sums.size() != p.numJobs * stride) throw std::invalid_argument("gpu::summarise: read-back has the wrong size");
  const float* s = out.sums.data() + job * stride;
  ClassroomResult r;
  r.learnerCount = static_cast<int>(p.numLearners);
  r.lastSessionDay = p.lastSessionDay;
  r.delayedTestDay = p.delayedTestDay;
  const double nan = std::nan("");
  for (auto* prof : {&r.score, &r.recall}) prof->days = p.days;
  for (std::size_t k = 0; k < p.numDays(); ++k) {
    const float* d = s + k * kDayStats;
    r.score.mean.push_back(d[0]);
    r.score.atRisk.push_back(d[3]);
    r.recall.mean.push_back(d[2]);
    r.recall.atRisk.push_back(nan);
    const bool q = p.quantiles && out.quantiles.size() == p.numJobs * p.numDays() * 3;
    const float* qs = q ? out.quantiles.data() + (job * p.numDays() + k) * 3 : nullptr;
    r.score.q10.push_back(q ? qs[0] : nan);
    r.score.q50.push_back(q ? qs[1] : nan);
    r.score.q90.push_back(q ? qs[2] : nan);
    r.recall.q10.push_back(nan);
    r.recall.q50.push_back(nan);
    r.recall.q90.push_back(nan);
  }
  if (p.profiles) {
    auto& tr = r.sessions;
    for (std::size_t i = 0; i < p.sessions; ++i) {
      const float* x = s + p.sessionOffset() + i * kSessionStats;
      tr.day.push_back(p.params[15] * static_cast<double>(i));
      tr.ability.push_back(x[0]);
      tr.effort.push_back(x[1]);
      tr.motivation.push_back(x[2]);
      tr.value.push_back(x[3]);
      tr.efficacy.push_back(x[4]);
      tr.anxiety.push_back(x[5]);
      tr.readiness.push_back(x[6]);
      tr.growth.push_back(x[7]);
      tr.engaged.push_back(x[8]);
      tr.zpdShare.push_back(x[9]);
    }
  }
  const float* j = s + p.jobOffset();
  r.endScore = j[0];
  r.delayedScore = j[1];
  r.abilityGain = j[2];
  const double n = static_cast<double>(p.numLearners);
  const double highCount = j[4] * n, lowCount = n - highCount;
  const double highSum = j[3] * n, lowSum = j[1] * n - highSum;
  r.equityGap = (highCount > 0 ? highSum / highCount : 0.0) - (lowCount > 0 ? lowSum / lowCount : 0.0);
  r.atRiskDelayed = s[p.testDayIndex * kDayStats + 3];
  r.retention = r.endScore > 0.0 ? r.delayedScore / r.endScore : 0.0;
  return r;
}

std::vector<double> delayedScores(const FusedPlan& p, const FusedOutput& out) {
  const std::size_t stride = p.statsPerJob();
  if (out.sums.size() != p.numJobs * stride) throw std::invalid_argument("gpu::delayedScores: read-back has the wrong size");
  std::vector<double> v;
  for (std::size_t job = 0; job < p.numJobs; ++job) v.push_back(out.sums[job * stride + p.jobOffset() + 1]);
  return v;
}

}  // namespace syn::gpu
