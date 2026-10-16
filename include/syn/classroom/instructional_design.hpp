#pragma once

// An instructional design: the teaching choices the chapter's theories recommend, each a lever
// in [0, 1]. The theories become presets (combinations of levers), and Gagné's nine events of
// instruction map onto the levers too, so every theory can be run through the same classroom
// engine and compared on the same learners.

#include <array>
#include <string>

namespace syn {

enum class Lever {
  Reinforcement = 0,  // behaviourism: points, rewards and contingent praise
  TeacherDirection,   // transmission of content by the teacher (the "banking" lecture)
  Empathy,            // humanism: supportive climate, normalising anxiety
  Choice,             // humanism / andragogy: self-direction, choice of topic and activity
  Relevance,          // humanism / andragogy / motivation: links to learners' lives and roles
  Transparency,       // Gagné "inform objectives", andragogy "need to know why"
  PriorActivation,    // Gagné "stimulate recall", constructivism: build on what learners know
  RetrievalPractice,  // cognitivism: quizzes and flash cards instead of rereading
  Spacing,            // cognitivism: return to earlier material at intervals
  Interleaving,       // cognitivism: mix related problem types
  Feedback,           // Gagné "provide feedback", corrective feedback after errors
  ProblemFirst,       // constructivism: productive struggle before instruction (disequilibrium)
  Scaffolding,        // Vygotsky: tasks targeted at each learner's ZPD, with support
  PeerDialogue,       // social constructivism: group work and talk
  Inclusivity,        // Bourdieu: culturally responsive, inclusive teaching
  GrowthFeedback,     // Dweck: praise process, "not yet"
};
constexpr int kLevers = 16;

const char* leverName(Lever l);
const char* leverTheory(Lever l);

struct InstructionalDesign {
  std::array<double, kLevers> lever{};

  double operator[](Lever l) const { return lever[static_cast<int>(l)]; }
  double& operator[](Lever l) { return lever[static_cast<int>(l)]; }
  /** Every lever clamped into [0, 1]. */
  InstructionalDesign clamped() const;
};

enum class Theory { Banking = 0, Behaviorism, Humanism, Cognitivism, Constructivism, SocialConstructivism, Andragogy, Blended };
constexpr int kTheories = 8;
const char* theoryName(Theory t);

/** The lesson each theory would design (the chapter's Activity 3.3 in lever form). */
InstructionalDesign theoryDesign(Theory t);

enum class GagneEvent {
  GainAttention = 0, InformObjectives, StimulateRecall, PresentStimulus, ProvideGuidance, ElicitPerformance,
  ProvideFeedback, AssessPerformance, EnhanceRetention
};
constexpr int kGagneEvents = 9;
const char* gagneEventName(GagneEvent e);

/** The design implied by a lesson that includes each of Gagné's events with the given intensity. */
InstructionalDesign designFromGagne(const std::array<double, kGagneEvents>& intensity);

}  // namespace syn
