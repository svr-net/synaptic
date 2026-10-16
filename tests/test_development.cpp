#include <numeric>

#include "syn/development/perry.hpp"
#include "syn/development/piaget.hpp"
#include "test_framework.hpp"

using namespace syn;

TEST(piaget_nominal_stages) {
  CHECK(nominalStage(1.0) == PiagetStage::Sensorimotor);
  CHECK(nominalStage(4.0) == PiagetStage::Preoperational);
  CHECK(nominalStage(9.0) == PiagetStage::ConcreteOperational);
  CHECK(nominalStage(15.0) == PiagetStage::FormalOperational);
  CHECK(stageProvides(PiagetStage::FormalOperational, Capability::Language));  // stages are cumulative
  CHECK(!stageProvides(PiagetStage::Preoperational, Capability::OtherPerspectives));
  CHECK(stageProvides(PiagetStage::ConcreteOperational, Capability::AppearanceVsReality));
  CHECK(!stageProvides(PiagetStage::ConcreteOperational, Capability::AbstractThinking));
  CHECK_THROWS(nominalStage(-1.0));
}

TEST(stage_distribution_by_age) {
  for (double age = 0.0; age <= 20.0; age += 0.5) {
    const auto p = stageDistribution(age);
    CHECK_NEAR(std::accumulate(p.begin(), p.end(), 0.0), 1.0, 1e-12);
  }
  CHECK(stageDistribution(4.5)[1] > 0.95);
  CHECK(stageDistribution(16.0)[3] > 0.99);
  CHECK_NEAR(stageDistribution(12.0)[3], 0.5, 1e-9);  // half the 12-year-olds have reached formal operations
}

TEST(climate_change_lesson_fits_older_children) {
  const auto lesson = climateChangeLesson();
  CHECK(lesson.size() == 6);
  for (const auto& a : lesson)
    for (double age = 3.0; age < 18.0; age += 1.0) CHECK(activityFit(a, age + 1.0) >= activityFit(a, age) - 1e-12);
  CHECK(activityFit(lesson[0], 5.0) > 0.95);  // habitat pictures work for preoperational children
  CHECK(activityFit(lesson[4], 5.0) < 0.01);  // anticipating consequences does not
  CHECK(activityFit(lesson[4], 16.0) > 0.99);
  CHECK(activityFit(lesson[3], 9.0) > 0.9);   // other perspectives from concrete operations
}

TEST(perry_progression_is_a_markov_chain) {
  PerryParams p;
  const auto path = perryProgression({1, 0, 0, 0}, p, 12);
  CHECK(path.size() == 13);
  for (const auto& d : path) CHECK_NEAR(d[0] + d[1] + d[2] + d[3], 1.0, 1e-12);
  CHECK_NEAR(path[1][1], p.transition[0], 1e-12);
  for (std::size_t t = 1; t < path.size(); ++t) CHECK(path[t][3] >= path[t - 1][3]);  // commitment is absorbing
  PerryParams challenged = p;
  challenged.challenge = 1.0;
  const auto fast = perryProgression({1, 0, 0, 0}, challenged, 12);
  CHECK(fast.back()[2] + fast.back()[3] > path.back()[2] + path.back()[3] + 0.05);
  CHECK_THROWS(perryProgression({0.5, 0, 0, 0}, p, 1));
}

TEST(relativists_judge_sources_by_evidence) {
  const double dual = sourceEvaluationAccuracy(PerryPosition::Dualism, 8, 300, 0.3, 1);
  const double multi = sourceEvaluationAccuracy(PerryPosition::Multiplicity, 8, 300, 0.3, 1);
  const double rel = sourceEvaluationAccuracy(PerryPosition::Relativism, 8, 300, 0.3, 1);
  const double com = sourceEvaluationAccuracy(PerryPosition::Commitment, 8, 300, 0.3, 1);
  CHECK_NEAR(multi, 0.5, 0.05);  // every view equally valid: no better than chance
  CHECK(dual > multi);           // authority is a weak but real cue
  CHECK(rel > dual + 0.15);
  CHECK(com >= rel - 0.01);
  // If authority were a perfect cue, the dualist would do well too.
  CHECK(sourceEvaluationAccuracy(PerryPosition::Dualism, 8, 300, 1.0, 1) > 0.85);
}
