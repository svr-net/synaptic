#include "syn/cognitivism/memory.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <stdexcept>

#include "syn/core/math.hpp"
#include "syn/core/random.hpp"

namespace syn {

const char* studyModeName(StudyMode m) {
  switch (m) {
    case StudyMode::Restudy: return "restudy";
    case StudyMode::Retrieval: return "retrieval";
    case StudyMode::RetrievalWithFeedback: return "retrieval + feedback";
  }
  return "?";
}

double MemoryTrace::retrievability(double t) const {
  return t <= lastStudy ? 1.0 : 1.0 / std::sqrt(1.0 + (19.0 / 81.0) * (t - lastStudy) / stability);
}

MemoryModel::MemoryModel(MemoryParams params) : params_(params) {
  if (params_.initialStability <= 0.0 || params_.minStability <= 0.0) throw std::invalid_argument("MemoryParams: stabilities must be positive");
  if (params_.failureFactor <= 0.0 || params_.failureFactor > 1.0) throw std::invalid_argument("MemoryParams: failureFactor must lie in (0, 1]");
}

MemoryTrace MemoryModel::encode(double t, double depth) const {
  if (depth <= 0.0) throw std::invalid_argument("MemoryModel::encode: depth must be positive");
  MemoryTrace m;
  m.stability = params_.initialStability * depth;
  m.lastStudy = t;
  return m;
}

double MemoryModel::gain(StudyMode mode, double r) const {
  if (mode == StudyMode::Restudy) return 1.0 + params_.restudyFloor + params_.restudyGain * (1.0 - r);
  return 1.0 + params_.retrievalFloor + params_.retrievalGain * (1.0 - r);
}

bool MemoryModel::study(MemoryTrace& m, double t, StudyMode mode, double u) const {
  if (t < m.lastStudy) throw std::invalid_argument("MemoryModel::study: events must be in time order");
  const double r = m.retrievability(t);
  ++m.reviews;
  if (mode == StudyMode::Restudy || u < r) {
    m.stability *= gain(mode, r);
    m.lastStudy = t;
    ++m.successes;
    return true;
  }
  if (mode == StudyMode::RetrievalWithFeedback) {
    // Failed attempt, then the answer is shown: the item is relearned from a weaker base.
    m.stability = std::max(params_.minStability, params_.failureFactor * m.stability) * gain(StudyMode::Restudy, r);
    m.lastStudy = t;
  }
  return false;
}

StudyPlan massedPlan(StudyMode mode, int count, double day) {
  StudyPlan p;
  std::ostringstream os;
  os << "massed " << studyModeName(mode) << " ×" << count << " on day " << day;
  p.name = os.str();
  for (int i = 0; i < count; ++i) p.reviews.push_back({day + 0.002 * (i + 1), mode});
  return p;
}

StudyPlan spacedPlan(StudyMode mode, int count, double gap) {
  StudyPlan p;
  std::ostringstream os;
  os << "spaced " << studyModeName(mode) << " ×" << count << " every " << gap << "d";
  p.name = os.str();
  for (int i = 0; i < count; ++i) p.reviews.push_back({gap * (i + 1), mode});
  return p;
}

StudyPlan expandingPlan(StudyMode mode, int count, double firstGap, double ratio) {
  StudyPlan p;
  std::ostringstream os;
  os << "expanding " << studyModeName(mode) << " ×" << count << " from " << firstGap << "d";
  p.name = os.str();
  double t = 0.0, gap = firstGap;
  for (int i = 0; i < count; ++i) {
    t += gap;
    gap *= ratio;
    p.reviews.push_back({t, mode});
  }
  return p;
}

RetentionResult simulatePlan(const MemoryModel& model, const StudyPlan& plan, double testDay, double horizon, int learners,
                             double heterogeneity, std::uint64_t seed) {
  if (learners < 1 || horizon <= 0.0) throw std::invalid_argument("simulatePlan: bad sizes");
  RetentionResult r;
  r.name = plan.name;
  const int steps = static_cast<int>(std::ceil(horizon));
  for (int d = 0; d <= steps; ++d) r.days.push_back(d);
  std::vector<std::vector<double>> recall(r.days.size(), std::vector<double>(learners));
  std::vector<StudyEvent> events = plan.reviews;
  std::sort(events.begin(), events.end(), [](const StudyEvent& a, const StudyEvent& b) { return a.day < b.day; });

  std::size_t attempts = 0, successes = 0;
  for (int i = 0; i < learners; ++i) {
    Rng rng = Rng::stream(seed, static_cast<std::uint64_t>(i));
    const double depth = plan.encodingDepth * std::exp(heterogeneity * rng.normal());
    std::vector<double> draws(events.size());
    for (auto& u : draws) u = rng.uniform();
    MemoryTrace m = model.encode(0.0, depth);
    std::size_t next = 0;
    for (std::size_t d = 0; d < r.days.size(); ++d) {
      const double day = r.days[d];
      while (next < events.size() && events[next].day <= day) {
        const bool ok = model.study(m, events[next].day, events[next].mode, draws[next]);
        if (events[next].mode != StudyMode::Restudy) {
          ++attempts;
          successes += ok ? 1 : 0;
        }
        ++next;
      }
      recall[d][i] = m.retrievability(day);
    }
    // The test: replay the events up to the test day with the same draws.
    MemoryTrace atTest = model.encode(0.0, depth);
    for (std::size_t k = 0; k < events.size() && events[k].day <= testDay; ++k)
      model.study(atTest, events[k].day, events[k].mode, draws[k]);
    r.recallAtTest += atTest.retrievability(testDay) / learners;
    r.meanStabilityAtTest += atTest.stability / learners;
  }
  for (const auto& v : recall) {
    r.meanRecall.push_back(mean(v));
    r.q10Recall.push_back(quantile(v, 0.1));
    r.q90Recall.push_back(quantile(v, 0.9));
  }
  r.reviewSuccessRate = attempts ? static_cast<double>(successes) / static_cast<double>(attempts) : 1.0;
  return r;
}

}  // namespace syn
