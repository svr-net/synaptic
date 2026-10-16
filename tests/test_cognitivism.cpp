#include <cmath>

#include "syn/cognitivism/interleaving.hpp"
#include "syn/cognitivism/memory.hpp"
#include "syn/cognitivism/schema.hpp"
#include "test_framework.hpp"

using namespace syn;

TEST(forgetting_curve) {
  MemoryModel m;
  const MemoryTrace t = m.encode(0.0, 2.0);
  CHECK_NEAR(t.stability, 2.0, 1e-12);
  CHECK_NEAR(t.retrievability(0.0), 1.0, 1e-12);
  CHECK_NEAR(t.retrievability(2.0), 0.9, 1e-12);  // S is the 90% interval
  for (double d = 1; d < 60; d += 1) CHECK(t.retrievability(d) < t.retrievability(d - 1));
  CHECK(t.retrievability(365) > 0.0);
}

TEST(study_events_update_stability) {
  MemoryModel m;
  const auto& p = m.params();
  MemoryTrace t = m.encode(0.0);
  const double r = t.retrievability(3.0);
  CHECK(m.study(t, 3.0, StudyMode::Restudy, 0.99));
  CHECK_NEAR(t.stability, 1.0 * (1 + p.restudyFloor + p.restudyGain * (1 - r)), 1e-12);
  MemoryTrace q = m.encode(0.0);
  CHECK(m.study(q, 3.0, StudyMode::Retrieval, 0.0));  // u < R: success
  CHECK_NEAR(q.stability, 1.0 * (1 + p.retrievalFloor + p.retrievalGain * (1 - r)), 1e-12);
  CHECK(q.stability > t.stability);  // the testing effect
  MemoryTrace f = m.encode(0.0);
  CHECK(!m.study(f, 3.0, StudyMode::Retrieval, 0.999));  // failure without feedback: nothing changes
  CHECK_NEAR(f.stability, 1.0, 1e-12);
  CHECK_NEAR(f.lastStudy, 0.0, 1e-12);
  MemoryTrace g = m.encode(0.0);
  CHECK(!m.study(g, 3.0, StudyMode::RetrievalWithFeedback, 0.999));  // feedback: relearned
  CHECK_NEAR(g.lastStudy, 3.0, 1e-12);
  CHECK(g.retrievability(10.0) > f.retrievability(10.0));
  CHECK_THROWS(m.study(g, 1.0, StudyMode::Restudy, 0.5));
}

TEST(testing_effect_and_spacing_effect) {
  MemoryModel m;
  auto at30 = [&](const StudyPlan& p) { return simulatePlan(m, p, 30, 40, 400, 0.3, 21).recallAtTest; };
  const double massedRestudy = at30(massedPlan(StudyMode::Restudy, 4, 0));
  const double massedRetrieval = at30(massedPlan(StudyMode::Retrieval, 4, 0));
  const double spacedRestudy = at30(spacedPlan(StudyMode::Restudy, 4, 2));
  const double spacedRetrieval = at30(spacedPlan(StudyMode::RetrievalWithFeedback, 4, 2));
  CHECK(massedRetrieval > massedRestudy);  // retrieval beats rereading
  CHECK(spacedRestudy > massedRestudy);    // spacing beats massing
  CHECK(spacedRetrieval > spacedRestudy);
  CHECK(spacedRetrieval > massedRetrieval);
  CHECK(at30(expandingPlan(StudyMode::RetrievalWithFeedback, 4, 1, 2)) > massedRetrieval);
}

TEST(cramming_works_for_tomorrow_not_next_month) {
  MemoryModel m;
  const auto cram = simulatePlan(m, massedPlan(StudyMode::Restudy, 4, 13), 30, 40, 400, 0.3, 4);
  const auto spaced = simulatePlan(m, spacedPlan(StudyMode::RetrievalWithFeedback, 4, 3), 30, 40, 400, 0.3, 4);
  CHECK(cram.meanRecall[14] > 0.9);              // the day after cramming
  CHECK(cram.recallAtTest < spaced.recallAtTest);  // a month later
  CHECK(cram.recallAtTest < cram.meanRecall[14] - 0.3);
  for (std::size_t d = 0; d < cram.days.size(); ++d) CHECK(cram.q10Recall[d] <= cram.meanRecall[d] + 1e-12 && cram.meanRecall[d] <= cram.q90Recall[d] + 1e-12);
}

TEST(interleaving_hurts_practice_but_helps_the_test) {
  const auto r = simulateInterleaving({}, 8);
  CHECK(r.blocked.practiceMean > r.interleaved.practiceMean);
  CHECK(r.interleaved.testAccuracy > r.blocked.testAccuracy + 0.2);
  CHECK_NEAR(r.blocked.execution, r.interleaved.execution, 1e-12);  // same amount of practice per type
  CHECK(r.interleaved.choiceAccuracy > r.blocked.choiceAccuracy);
}

TEST(schema_spreading_activation) {
  auto s = SchemaNetwork::librarySearchExample();
  const auto q = s.index("research question");
  const auto a = s.spread({q});
  CHECK_NEAR(a[q], 1.0, 1e-12);
  CHECK(a[s.index("keyword")] > a[s.index("synonym")]);  // activation decays with distance
  CHECK_NEAR(a[s.index("Venn diagram")], 0.0, 1e-12);   // not connected within three steps
  CHECK(s.recallProbability(s.index("keyword"), {q}) > s.recallProbability(s.index("search engine"), {q}));
  CHECK_THROWS(s.addConcept("keyword"));
}

TEST(stimulating_prior_recall_gives_more_retrieval_routes) {
  auto plain = SchemaNetwork::librarySearchExample();
  auto primed = SchemaNetwork::librarySearchExample();
  const auto venn = plain.index("Venn diagram");
  const auto andPlain = plain.learn("AND operator", {venn}, {});
  const auto andPrimed = primed.learn("AND operator", {venn},
                                      {primed.index("narrow results"), primed.index("keyword"), primed.index("research question")});
  CHECK(primed.degree(andPrimed) > plain.degree(andPlain));
  // Later, a learner who starts from their research question can find AND only via the links.
  const auto q = plain.index("research question");
  CHECK(primed.recallProbability(andPrimed, {q}) > plain.recallProbability(andPlain, {q}) + 0.3);
}
