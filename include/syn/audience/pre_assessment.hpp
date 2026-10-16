#pragma once

// Identifying audience needs (chapter 7): what the instructor knows about a class before a
// one-shot session, and what that knowledge is worth.
//
// A community contains many classes. Class c has a mean prior knowledge μ_c ~ N(μ, τ²) and its
// learners θ ~ N(μ_c, σ²). The instructor must decide where to pitch a 50-minute one-shot
// session. A one-shot teaches one specific skill, so it is sharper than a course: its success
// depends on (b − θ)·k, k the sharpness (a session that starts with job databases fails learners
// who cannot yet find a job bank). It is pitched best at the ZPD's peak, b = μ_c + s/(2k).
//
// 1. Audience research and community data give a prior belief about the class, N(μ + bias, s₀²):
//    no information (an assumption, e.g. "they have already chosen a topic"), research
//    literature (Project Information Literacy, Pew: broad, may not generalise), demographics and
//    campus or community offices (aggregate data about this community), or a curriculum
//    framework / the course instructor (the school librarian's advantage).
// 2. A pre-assessment measures the learners themselves. Each instrument costs class time and
//    analysis time and has its own measurement properties: noise, a recall bias (knowing the
//    names of the Boolean operators is not understanding how to use them), a social-desirability
//    bias when answers are not anonymous, participation, whether it yields individual results
//    (needed to group learners), whether it surfaces worries (affective information), builds
//    interest and previews the lesson (engagement), and reveals learners' interests.
// 3. The instructor combines prior and measurement by precision weighting (the normal posterior),
//    pitches the session at the estimate, and with individual results may differentiate:
//    group learners by their pre-assessment score and pitch each group separately.
// Open-ended instruments (K-W-L, concept maps) take longer to analyse; in a large class only the
// part that can be read in the analysis budget informs the estimate.

#include <cstdint>
#include <string>
#include <vector>

namespace syn {

struct PriorSource {
  std::string name;
  double bias = 0.0;  // systematic error of the belief about the class mean (logits)
  double sd = 1.0;    // the instructor's uncertainty s₀
  static PriorSource assumption();     // no information: assumes too much
  static PriorSource research();       // research literature and reports
  static PriorSource demographics();   // census, campus and community offices
  static PriorSource curriculum();     // curriculum framework, the course instructor
  static std::vector<PriorSource> all();
};

struct PreAssessment {
  std::string name;
  double classMinutes = 0.0;        // class time the learners spend on it
  double analysisPerLearner = 0.0;  // minutes to read one response
  double noise = 1.0;               // sd of one response around the learner's understanding
  double recallBias = 0.0;          // overestimate from measuring recall, not understanding
  double socialBias = 0.0;          // overestimate when answers are public
  double participation = 1.0;       // share of learners who respond
  bool individual = false;          // identifiable individual results (needed for grouping)
  double affective = 0.0;           // surfaces worries, [0, 1]
  double engagement = 0.0;          // previews the lesson, builds interest, [0, 1]
  double interests = 0.0;           // reveals learners' interests and examples, [0, 1]
  double anxiety = 0.0;             // anxiety it causes (high-stakes formats), [0, 1]

  static PreAssessment none();
  static PreAssessment showOfHands();
  static PreAssessment anonymousPoll();
  static PreAssessment worksheet();
  static PreAssessment kwl();
  static PreAssessment conceptMap();
  static PreAssessment brainstorm();
  static PreAssessment quiz();
  static PreAssessment homework();
  static std::vector<PreAssessment> all();
};

struct OneShotParams {
  int classes = 400;              // classes simulated (Monte Carlo)
  int classSize = 25;
  double communityMean = 0.0;     // μ
  double betweenClassSd = 0.6;    // τ
  double withinClassSd = 0.9;     // σ
  double anxietyMean = 0.4;       // class anxiety about the research task
  double sessionMinutes = 50.0;
  double analysisBudget = 3.0;    // minutes the instructor can spend reading responses in class
  double scaffold = 1.5;          // support during the session (logits of the skill)
  double sharpness = 2.0;         // k: skill logits per logit of prior knowledge
  int groups = 1;                 // differentiation: groups by pre-assessment score (needs individual results)
  double groupCost = 0.05;        // teaching efficiency lost per extra group (attention split)
};

struct OneShotResult {
  std::string prior, instrument;
  double meanGain = 0.0;          // mean learning gain per learner
  double zpdShare = 0.0;          // share of learners whose task was in their ZPD
  double estimateRmse = 0.0;      // error of the estimated class mean
  double pitchError = 0.0;        // mean |pitched level − ideal level|
  double teachingMinutes = 0.0;
  double analysedShare = 0.0;     // share of responses read in the analysis budget
  double gainWithoutPreAssessment = 0.0;  // same classes, same prior, no instrument
  double valueOfInformation = 0.0;        // meanGain − gainWithoutPreAssessment
};

OneShotResult simulateOneShot(const OneShotParams& params, const PriorSource& prior, const PreAssessment& instrument,
                              std::uint64_t seed);

}  // namespace syn
