#include "syn/classroom/instructional_design.hpp"

#include "syn/core/math.hpp"

namespace syn {

const char* leverName(Lever l) {
  static const char* names[kLevers] = {
      "reinforcement",      "teacher direction", "empathy",          "choice",       "relevance",     "transparency",
      "prior activation",   "retrieval practice", "spacing",         "interleaving", "feedback",      "problem first",
      "scaffolding (ZPD)",  "peer dialogue",      "inclusivity",     "growth feedback"};
  const int k = static_cast<int>(l);
  return k >= 0 && k < kLevers ? names[k] : "?";
}

const char* leverTheory(Lever l) {
  switch (l) {
    case Lever::Reinforcement: return "behaviourism";
    case Lever::TeacherDirection: return "behaviourism / cognitivism";
    case Lever::Empathy: return "humanism";
    case Lever::Choice: return "humanism / andragogy";
    case Lever::Relevance: return "andragogy / motivation";
    case Lever::Transparency: return "Gagné / andragogy";
    case Lever::PriorActivation: return "constructivism / Gagné";
    case Lever::RetrievalPractice:
    case Lever::Spacing:
    case Lever::Interleaving: return "cognitivism";
    case Lever::Feedback: return "Gagné / motivation";
    case Lever::ProblemFirst: return "constructivism";
    case Lever::Scaffolding: return "Vygotsky (ZPD)";
    case Lever::PeerDialogue: return "social constructivism";
    case Lever::Inclusivity: return "Bourdieu / critical practice";
    case Lever::GrowthFeedback: return "growth mindset";
  }
  return "?";
}

InstructionalDesign InstructionalDesign::clamped() const {
  InstructionalDesign d = *this;
  for (double& x : d.lever) x = clamp01(x);
  return d;
}

const char* theoryName(Theory t) {
  static const char* names[kTheories] = {"banking lecture", "behaviourism", "humanism", "cognitivism",
                                         "constructivism", "social constructivism", "andragogy", "blended"};
  const int k = static_cast<int>(t);
  return k >= 0 && k < kTheories ? names[k] : "?";
}

InstructionalDesign theoryDesign(Theory t) {
  InstructionalDesign d;
  auto set = [&](Lever l, double v) { d[l] = v; };
  switch (t) {
    case Theory::Banking:
      set(Lever::TeacherDirection, 1.0);
      break;
    case Theory::Behaviorism:
      set(Lever::TeacherDirection, 1.0);
      set(Lever::Reinforcement, 1.0);
      set(Lever::Feedback, 0.6);
      set(Lever::RetrievalPractice, 0.3);  // drill: mostly repetition
      break;
    case Theory::Humanism:
      set(Lever::TeacherDirection, 0.2);
      set(Lever::Empathy, 1.0);
      set(Lever::Choice, 0.8);
      set(Lever::Relevance, 0.9);
      set(Lever::Inclusivity, 0.5);
      set(Lever::GrowthFeedback, 0.3);
      break;
    case Theory::Cognitivism:
      set(Lever::TeacherDirection, 0.8);
      set(Lever::Transparency, 1.0);
      set(Lever::PriorActivation, 1.0);
      set(Lever::RetrievalPractice, 1.0);
      set(Lever::Spacing, 1.0);
      set(Lever::Interleaving, 1.0);
      set(Lever::Feedback, 1.0);
      set(Lever::ProblemFirst, 0.4);
      set(Lever::Scaffolding, 0.3);
      break;
    case Theory::Constructivism:
      set(Lever::TeacherDirection, 0.2);
      set(Lever::ProblemFirst, 1.0);
      set(Lever::PriorActivation, 1.0);
      set(Lever::Empathy, 0.6);
      set(Lever::Choice, 0.4);
      set(Lever::Relevance, 0.6);
      set(Lever::Scaffolding, 0.4);
      break;
    case Theory::SocialConstructivism:
      set(Lever::TeacherDirection, 0.3);
      set(Lever::PeerDialogue, 1.0);
      set(Lever::Scaffolding, 1.0);
      set(Lever::Inclusivity, 1.0);
      set(Lever::PriorActivation, 0.5);
      set(Lever::ProblemFirst, 0.4);
      set(Lever::Empathy, 0.5);
      break;
    case Theory::Andragogy:
      set(Lever::TeacherDirection, 0.2);
      set(Lever::Choice, 1.0);
      set(Lever::Relevance, 1.0);
      set(Lever::Transparency, 1.0);
      set(Lever::PriorActivation, 0.8);  // experience as a resource
      set(Lever::Empathy, 0.5);
      set(Lever::ProblemFirst, 0.5);     // problem- and performance-centred
      break;
    case Theory::Blended:
      d.lever = {0.3, 0.5, 0.8, 0.5, 0.8, 1.0, 0.8, 0.9, 0.9, 0.7, 1.0, 0.5, 0.8, 0.7, 0.8, 0.8};
      break;
  }
  return d;
}

const char* gagneEventName(GagneEvent e) {
  static const char* names[kGagneEvents] = {
      "gain attention", "inform the learner of the objective", "stimulate recall of prior learning",
      "present the stimulus", "provide learner guidance", "elicit performance", "provide feedback",
      "assess performance", "enhance retention and transfer"};
  const int k = static_cast<int>(e);
  return k >= 0 && k < kGagneEvents ? names[k] : "?";
}

InstructionalDesign designFromGagne(const std::array<double, kGagneEvents>& x) {
  auto at = [&](GagneEvent e) { return clamp01(x[static_cast<int>(e)]); };
  InstructionalDesign d;
  d[Lever::Relevance] = 0.5 * at(GagneEvent::GainAttention);
  d[Lever::Transparency] = at(GagneEvent::InformObjectives);
  d[Lever::PriorActivation] = at(GagneEvent::StimulateRecall);
  d[Lever::TeacherDirection] = at(GagneEvent::PresentStimulus);
  d[Lever::Scaffolding] = 0.6 * at(GagneEvent::ProvideGuidance) + 0.3 * at(GagneEvent::AssessPerformance);
  d[Lever::RetrievalPractice] = 0.8 * at(GagneEvent::ElicitPerformance) + 0.2 * at(GagneEvent::AssessPerformance);
  d[Lever::Feedback] = at(GagneEvent::ProvideFeedback);
  d[Lever::Spacing] = at(GagneEvent::EnhanceRetention);
  d[Lever::Interleaving] = 0.6 * at(GagneEvent::EnhanceRetention);
  return d.clamped();
}

}  // namespace syn
