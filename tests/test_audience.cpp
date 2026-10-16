#include <cmath>

#include "syn/audience/community.hpp"
#include "syn/audience/pre_assessment.hpp"
#include "syn/humanism/information_search.hpp"
#include "test_framework.hpp"

using namespace syn;

TEST(pre_assessment_pays_when_the_prior_is_poor) {
  OneShotParams p;
  const auto assumed = PriorSource::assumption();
  const auto none = simulateOneShot(p, assumed, PreAssessment::none(), 7);
  CHECK_NEAR(none.valueOfInformation, 0.0, 1e-12);
  CHECK_NEAR(none.teachingMinutes, p.sessionMinutes, 1e-12);
  CHECK_NEAR(none.analysedShare, 0.0, 1e-12);
  for (const auto& inst : {PreAssessment::anonymousPoll(), PreAssessment::kwl(), PreAssessment::worksheet()}) {
    const auto r = simulateOneShot(p, assumed, inst, 7);
    CHECK(r.valueOfInformation > 0.005);  // worth its class time
    CHECK(r.estimateRmse < none.estimateRmse);
    CHECK_NEAR(r.gainWithoutPreAssessment, none.meanGain, 1e-12);  // same classes, same learners
  }
  // A long formal quiz costs more class time than its information is worth.
  CHECK(simulateOneShot(p, assumed, PreAssessment::quiz(), 7).valueOfInformation < 0.0);
}

TEST(better_audience_research_needs_less_pre_assessment) {
  OneShotParams p;
  const auto inst = PreAssessment::anonymousPoll();
  const double fromAssumption = simulateOneShot(p, PriorSource::assumption(), inst, 7).valueOfInformation;
  const double fromCurriculum = simulateOneShot(p, PriorSource::curriculum(), inst, 7).valueOfInformation;
  CHECK(fromCurriculum < fromAssumption);
  const auto none = [&](const PriorSource& s) { return simulateOneShot(p, s, PreAssessment::none(), 7).meanGain; };
  CHECK(none(PriorSource::curriculum()) > none(PriorSource::assumption()));
  CHECK(none(PriorSource::research()) > none(PriorSource::assumption()));
}

TEST(measurement_properties_of_instruments) {
  OneShotParams p;
  // Public answers and recall-only questions overestimate understanding: a worksheet that asks
  // learners to use the operators estimates the class better than a show of hands.
  const auto hands = simulateOneShot(p, PriorSource::research(), PreAssessment::showOfHands(), 3);
  const auto sheet = simulateOneShot(p, PriorSource::research(), PreAssessment::worksheet(), 3);
  CHECK(sheet.estimateRmse < hands.estimateRmse);
  CHECK(sheet.teachingMinutes < hands.teachingMinutes);
  // Open-ended responses cannot all be read in a large class.
  OneShotParams big = p;
  big.classSize = 120;
  const auto kwlSmall = simulateOneShot(p, PriorSource::research(), PreAssessment::kwl(), 3);
  const auto kwlBig = simulateOneShot(big, PriorSource::research(), PreAssessment::kwl(), 3);
  CHECK(kwlBig.analysedShare < kwlSmall.analysedShare);
  CHECK(simulateOneShot(big, PriorSource::research(), PreAssessment::anonymousPoll(), 3).analysedShare > 0.7);
}

TEST(grouping_helps_mixed_classes) {
  auto run = [](double sd, int groups) {
    OneShotParams p;
    p.withinClassSd = sd;
    p.groups = groups;
    return simulateOneShot(p, PriorSource::research(), PreAssessment::worksheet(), 7);
  };
  CHECK(run(1.4, 2).meanGain > run(1.4, 1).meanGain);  // a mixed class: differentiate
  CHECK(run(0.5, 2).meanGain < run(0.5, 1).meanGain);  // a homogeneous class: one level is enough
  CHECK(run(0.9, 3).zpdShare > run(0.9, 1).zpdShare);
  // A poll gives no individual results, so it cannot be used to group learners.
  OneShotParams p;
  p.groups = 3;
  OneShotParams one = p;
  one.groups = 1;
  CHECK(simulateOneShot(p, PriorSource::research(), PreAssessment::anonymousPoll(), 7).meanGain ==
        simulateOneShot(one, PriorSource::research(), PreAssessment::anonymousPoll(), 7).meanGain);
}

TEST(demographics_describe_groups_not_individuals) {
  const auto r = simulateCommunity({}, 3);
  CHECK(r.attributes.size() == defaultCommunityAttributes().size());
  CHECK(r.bestSingleR2 < 0.1);                 // a single label says little about an individual
  CHECK(r.intersectionR2 >= r.bestSingleR2);   // intersections say a little more
  CHECK(r.intersectionR2 < 0.2);
  CHECK(r.preAssessmentR2 > 0.6);              // measuring the learner says much more
  CHECK(r.combinedR2 >= r.preAssessmentR2 - 1e-9);
  CHECK_NEAR(r.groupShare[0], 0.7, 0.03);      // nontraditional students
  // Privacy: the bigger the minimum cell size, the more intersections are suppressed.
  for (std::size_t i = 1; i < r.suppressedCells.size(); ++i) {
    CHECK(r.suppressedCells[i] >= r.suppressedCells[i - 1]);
    CHECK(r.suppressedLearners[i] >= r.suppressedLearners[i - 1]);
  }
  CHECK_NEAR(r.suppressedCells[0], 0.0, 1e-12);
  CHECK(r.suppressedCells.back() > 0.3);
}

TEST(zones_of_intervention_and_the_seven_stages) {
  CHECK(kIspStages == 7);
  CHECK(std::string(ispStages()[6].name) == "assessment");
  IspParams none;
  const double base = simulateInformationSearch(none, 5).completedOnTime;
  auto at = [](IspStage s, int day) {
    IspParams p;
    p.interventionStage = s;
    p.interventionDay = day;
    return simulateInformationSearch(p, 5);
  };
  // Getting started is the hardest part (Project Information Literacy): a session on finding
  // background information and a focus helps more than one on searching for articles.
  CHECK(at(IspStage::Exploration, 7).completedOnTime > at(IspStage::Collection, 7).completedOnTime + 0.05);
  CHECK(at(IspStage::Collection, 7).completedOnTime > base);
  // A session for a stage the students have already left comes too late.
  const auto late = at(IspStage::Selection, 14);
  CHECK(late.helped < 0.2);
  CHECK(std::fabs(late.completedOnTime - base) < 0.03);
  CHECK_NEAR(simulateInformationSearch(none, 5).helped, 0.0, 1e-12);
}
