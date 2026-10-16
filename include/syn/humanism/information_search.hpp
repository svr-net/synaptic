#pragma once

// Kuhlthau's Information Search Process (ISP) with Mellon's library anxiety.
//
// A research assignment moves through six stages, each with a typical feeling: uncertainty at
// initiation, optimism at selection, confusion and doubt during exploration, clarity at
// formulation, confidence during collection, and satisfaction or disappointment at
// presentation. Anxiety interferes with the cognitive work of each stage: progress per day is
// effort / (1 + interference · uncertainty). A humanist (empathetic) instructor acknowledges
// that the feelings are normal and offers support, which lowers the uncertainty felt.

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace syn {

enum class IspStage { Initiation = 0, Selection, Exploration, Formulation, Collection, Presentation };
constexpr int kIspStages = 6;

struct IspStageInfo {
  const char* name;
  const char* feeling;
  double uncertainty;  // typical affective uncertainty in [0, 1]
  double work;         // effort-days needed at full efficiency
};

const std::array<IspStageInfo, kIspStages>& ispStages();

struct IspParams {
  int students = 400;
  int deadline = 42;               // days until the paper is due
  double anxietyMean = 0.5;        // library anxiety (trait) in [0, 1]
  double anxietySd = 0.2;
  double anxietyAmplification = 0.8;  // how strongly trait anxiety amplifies a stage's uncertainty
  double interference = 1.6;       // anxiety interference with cognitive work
  double effortMean = 1.0;
  double effortSd = 0.25;
  double support = 0.0;            // empathetic instruction in [0, 1]: normalising feelings, help
  double relevance = 0.0;          // relevance of the topic to the student in [0, 1]
};

struct IspResult {
  std::vector<double> days;
  std::vector<double> meanUncertainty;                 // per day, over students still working
  std::array<std::vector<double>, kIspStages + 1> stageShare;  // per stage (+ completed) per day
  std::array<double, kIspStages> meanDaysInStage{};
  double completedOnTime = 0.0;  // share that finished presentation by the deadline
  double satisfied = 0.0;        // finished with time to spare for collection (a focused paper)
  double meanCompletionDay = 0.0;
};

IspResult simulateInformationSearch(const IspParams& params, std::uint64_t seed);

}  // namespace syn
