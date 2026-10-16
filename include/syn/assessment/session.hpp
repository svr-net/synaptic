#pragma once

// Formative and summative assessment in a session, and the assessment cycle (chapter 9).
//
// A session teaches an outcome in chunks, each building on the one before. Learner i masters
// chunk j with the ZPD geometry (pure elliptic CGA, social/zpd.hpp): the chance of success with
// the session's support, P_assisted(θ_i + c_j + κ·(m_{i,j−1} − 1), b_j), where c_j is how clearly
// the chunk was taught (unknown to the instructor: "maybe some part of our lesson was unclear"),
// κ the weight of the prerequisite and the support scales with the minutes the chunk gets.
//
// Formative assessment ("taking the temperature of the class") checks each chunk before the next
// builds on it: an informal check is quick but flattering (learners nod along) and reaches only
// some learners; a formal one costs more time and is accurate. When a check shows the class has
// not got a chunk, the instructor reteaches it (clearer, and at the cost of later chunks' time).
// Summative assessment at the end costs time, helps no one in the room, but tells the
// instructor whether the outcome was met, and which chunks were not, for next time.
//
// The assessment cycle (Figure 9.1): set outcomes → choose activities → collect data → analyse →
// revisit outcomes and reflect on instruction, offering after offering. With direct evidence the
// instructor finds the unclear chunks and fixes them, makes an outcome most learners meet more
// challenging and one few meet more achievable. Self-report hides the gaps; grades (attendance,
// effort, conduct) say little about any chunk and flatter the class. Evidence per chunk is the
// share of learners meeting it (a worksheet item per chunk).

#include <cstdint>
#include <string>
#include <vector>

namespace syn {

enum class FormativeCheck { None = 0, Informal, Formal };
const char* formativeCheckName(FormativeCheck c);

struct AssessedSessionParams {
  int classes = 300;
  int classSize = 25;
  int chunks = 4;
  double minutes = 45.0;
  double betweenClassSd = 0.5;
  double withinClassSd = 0.9;
  double clarityMean = 0.0;        // how clearly the instructor teaches on average
  double claritySd = 0.8;          // chunk to chunk
  std::vector<double> chunkClarity;  // fixed clarity per chunk (the assessment cycle); empty: drawn per class
  double difficulty = 0.0;         // b of the last chunk (the outcome); earlier chunks are easier
  double scaffold = 1.5;           // support at the nominal time per chunk
  double prerequisite = 1.5;       // κ
  FormativeCheck formative = FormativeCheck::None;
  bool summative = false;
  double reteachThreshold = 0.7;   // reteach a chunk whose estimated mastery is below this
  double reteachMinutes = 4.0;
  double reteachClarity = 1.0;     // a second, clearer explanation
  double threshold = 0.6;          // a learner meets the outcome at this mastery
};

struct AssessedSessionResult {
  std::string name;
  std::vector<double> chunkMastery;  // mean mastery per chunk
  double finalMastery = 0.0;         // mean mastery of the outcome (last chunk)
  double outcomeMet = 0.0;           // share of learners meeting the outcome
  double reteaches = 0.0;            // chunks retaught per class
  double missedGaps = 0.0;           // chunks below the reteach threshold that the check missed, per class
  double assessmentMinutes = 0.0;    // class time on assessment per class
  double estimateError = 0.0;        // |instructor's estimate of the share met − true share|
};

AssessedSessionResult simulateAssessedSession(const AssessedSessionParams& params, std::uint64_t seed);

/** What the instructor learns after each offering. */
enum class CycleEvidence { None = 0, Grades, SelfReport, Direct };
const char* cycleEvidenceName(CycleEvidence e);

struct AssessmentCycleParams {
  int offerings = 8;
  AssessedSessionParams session;  // chunkClarity, if empty, is drawn once for the instructor
  CycleEvidence evidence = CycleEvidence::Direct;
  double revise = 0.5;           // clarity gained by revising a chunk found to be a gap
  double gapThreshold = 0.7;     // a chunk fewer than this share of learners meet is a gap
  double raiseAbove = 0.85;      // most meet it: make the outcome more challenging
  double lowerBelow = 0.4;       // few meet it: make it more achievable
  double step = 0.3;             // change of the outcome's difficulty
};

struct AssessmentCycleResult {
  std::string evidence;
  std::vector<double> met;          // per offering
  std::vector<double> mastery;      // mean mastery of the outcome per offering
  std::vector<double> difficulty;   // the outcome's difficulty per offering
  std::vector<double> clarity;      // mean chunk clarity per offering
  std::vector<double> estimatedMet; // what the evidence said
};

AssessmentCycleResult simulateAssessmentCycle(const AssessmentCycleParams& params, std::uint64_t seed);

}  // namespace syn
