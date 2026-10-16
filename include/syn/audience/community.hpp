#pragma once

// Learning about audiences through demographics and community data, and its limits.
//
// Census figures and campus or community offices describe groups: older adults, first-generation
// students, English language learners, nontraditional students (about 70% of undergraduates meet
// at least one criterion), learners with disabilities. People are intersectional: each learner
// has several of these attributes, and within any group individual needs vary widely. In the
// model a learner's prior knowledge is the sum of the effects of their attributes plus an
// individual deviation, so a group label predicts an individual only as well as the share of
// variance the attributes explain. Pre-assessment measures the individual directly.
//
// Privacy: librarians should ask campus and community offices for aggregate data only. Reporting
// by intersection of attributes quickly produces small cells that could identify individuals;
// cells with fewer than k members are suppressed (a minimum cell size, as FERPA-style practice
// recommends), and the model reports how much of the community those cells cover.

#include <cstdint>
#include <string>
#include <vector>

namespace syn {

struct CommunityAttribute {
  std::string name;
  double prevalence = 0.0;  // share of the community with the attribute
  double effect = 0.0;      // shift of prior knowledge of the topic (logits)
};

std::vector<CommunityAttribute> defaultCommunityAttributes();

struct CommunityParams {
  int learners = 3000;
  double individualSd = 0.9;  // within-group variation of prior knowledge
  double measurementNoise = 0.6;  // sd of a pre-assessment response (a worksheet)
  std::vector<CommunityAttribute> attributes = defaultCommunityAttributes();
};

struct CommunityResult {
  std::vector<std::string> attributes;
  std::vector<double> groupShare;     // observed share with each attribute
  std::vector<double> groupMean;      // mean prior knowledge of learners with each attribute
  std::vector<double> singleR2;       // variance explained by each attribute alone
  double bestSingleR2 = 0.0;          // "pigeon-holing" by the most telling attribute
  double intersectionR2 = 0.0;        // every combination of attributes (cell means)
  double preAssessmentR2 = 0.0;       // a pre-assessment response
  double combinedR2 = 0.0;            // cell mean and pre-assessment, precision weighted
  int cells = 0;                      // non-empty intersection cells
  std::vector<int> minCellSizes;      // k = 1, 5, 10, 20, 50
  std::vector<double> suppressedCells;     // share of non-empty cells with fewer than k members
  std::vector<double> suppressedLearners;  // share of learners in those cells
};

CommunityResult simulateCommunity(const CommunityParams& params, std::uint64_t seed);

}  // namespace syn
