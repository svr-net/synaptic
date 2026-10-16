#include "syn/constructivism/equilibration.hpp"
#include "test_framework.hpp"

using namespace syn;

TEST(assimilation_keeps_the_structure) {
  SchemaLearner l({});
  l.addSchema({1.0, 0.0, 0.0}, 0);
  CHECK(l.observe({{1.1, 0.1, 0.0}, 0}, 0.5) == Adaptation::Assimilation);
  CHECK(l.schemas().size() == 1);
  CHECK(l.prototypeFeatures(0)[0] > 1.0);  // rotated towards the observation
  CHECK(l.geometry().isPoint(l.schemas()[0].prototype));
  CHECK_NEAR(l.disequilibrium(), 0.0, 1e-12);
  // An unlabelled cat close enough to the dog prototype is called a dog.
  CHECK(l.observe({{0.3, 1.0, 1.0}, -1}, 0.5) == Adaptation::Assimilation);
  CHECK(l.classify({0.3, 1.0, 1.0}) == 0);
}

TEST(conflict_builds_disequilibrium_until_accommodation) {
  EquilibrationParams p;
  SchemaLearner l(p);
  l.addSchema({1.0, 0.0, 0.0}, 0);
  const Observation cat{{0.3, 1.0, 1.0}, 1};  // "that's a cat"
  CHECK(l.observe(cat, 0.5) == Adaptation::Disequilibrium);
  CHECK_NEAR(l.disequilibrium(), 1.0 - p.persistence, 1e-12);
  Adaptation a = Adaptation::Disequilibrium;
  int n = 1;
  while (a == Adaptation::Disequilibrium && n < 20) {
    a = l.observe(cat, 0.5);
    ++n;
  }
  CHECK(a == Adaptation::Accommodation);
  CHECK(l.schemas().size() == 2);
  CHECK(l.schemas()[1].label == 1);
  CHECK(l.classify({0.3, 1.0, 1.0}) == 1);
  CHECK(l.classify({1.0, 0.0, 0.0}) == 0);
  CHECK(l.disequilibrium() < p.accommodationThreshold);
}

TEST(avoiding_discomfort_avoids_learning) {
  EquilibrationParams avoid;
  avoid.avoidance = 1.0;
  SchemaLearner l(avoid);
  l.addSchema({1.0, 0.0, 0.0}, 0);
  for (int i = 0; i < 50; ++i) CHECK(l.observe({{0.3, 1.0, 1.0}, 1}, 0.5) == Adaptation::Avoidance);
  CHECK(l.schemas().size() == 1);
  CHECK_NEAR(l.disequilibrium(), 0.0, 1e-12);
}

TEST(child_learns_cats_only_with_feedback) {
  AnimalWorld silent;
  silent.feedback = 0.0;
  AnimalWorld named;
  named.feedback = 0.3;
  const auto a = simulateEquilibration({}, silent, 5), b = simulateEquilibration({}, named, 5);
  CHECK_NEAR(a.accuracy.front(), 0.5, 0.02);  // only a dog schema: every cat is called a dog
  CHECK(a.finalAccuracy < 0.6);
  CHECK(b.finalAccuracy > 0.9);
  CHECK(b.accommodations >= 0.9);
  CHECK(b.assimilations > 10 * b.accommodations);  // most experience is assimilated
  EquilibrationParams avoidant;
  avoidant.avoidance = 0.8;
  const auto c = simulateEquilibration(avoidant, named, 5);
  CHECK(c.finalAccuracy < b.finalAccuracy - 0.2);
  CHECK(c.avoidances > 0.0);
}
