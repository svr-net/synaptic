#pragma once

// Backward Design (Wiggins & McTighe 2005), chapter 8.
//
// Stage 1 identifies the desired results (learning outcomes), stage 2 the acceptable evidence
// (assessments, see bloom.hpp: checkEvidence), stage 3 the learning experiences. Planning from the
// outcomes avoids the "twin sins":
//   coverage-based instruction  packs every topic into the session (the 50-minute one-shot with
//                               the website tour, interlibrary loan, the chat link, the catalogue,
//                               a database, keyword vs subject searching, Boolean operators and
//                               nesting), mostly by lecture: much is touched, little is learned;
//   activity-based instruction  practises without a purpose: learners acquire basic skills but do
//                               not connect them to the big idea, so they do not transfer them and
//                               go back to their old search habits.
//
// The content model is Figure 8.3's three circles: big ideas (enduring understandings) rest on
// basic ideas (the knowledge and skills needed to reach them); outside both is what is worth
// being familiar with but not essential (Melvil Dewey, the term OPAC). A session plan gives
// minutes to items. A learner masters item i with m_i = 1 − exp(−e^a · r · t_i · η / T_i) where
// a is the learner's aptitude, t_i the minutes, T_i the minutes the item needs and η the
// efficiency of the teaching (lecture 0.45, active learning up to 1). Touching more items than
// working memory and attention can hold (≈4 in an hour; the chapter's "three to four outcomes in a
// 60-minute session") divides the efficiency by 1 + λ·max(0, n − c), and each item costs a
// transition. The understanding of a big idea is its own mastery times the mean mastery of its
// basic ideas (a big idea cannot be reached without its building blocks); transfer to new
// contexts needs the activities to be aligned with the outcomes.
//
// Shared outcomes set expectations and let learners assess themselves; essential questions and
// affective framing (Krathwohl's affective domain, Fink's caring and human dimension) make the
// relevance explicit. Both raise the share of learners who change their behaviour afterwards.

#include <cstdint>
#include <string>
#include <vector>

#include "syn/outcomes/bloom.hpp"

namespace syn {

enum class ContentKind { BigIdea = 0, Basic, NiceToKnow };
const char* contentKindName(ContentKind k);

struct ContentItem {
  std::string name;
  ContentKind kind = ContentKind::Basic;
  int bigIdea = -1;              // the big idea a basic idea builds (index into the items), −1 none
  BloomLevel level = BloomLevel::Remember;
  double minutesToMaster = 8.0;  // T_i
};

/** A one-shot library session's content (Tables 8.5 and 8.6, and the crammed session of the text). */
std::vector<ContentItem> librarySessionContent();

struct SessionPlan {
  std::string name;
  std::vector<double> minutes;    // per content item; 0 = not touched
  double activeShare = 0.0;       // share of time learners practise rather than listen
  double alignment = 0.0;         // activities linked to outcomes and big ideas, [0, 1]
  bool outcomesShared = false;    // outcomes stated to learners at the start
  double essentialQuestions = 0.0;  // relevance made explicit, [0, 1]
  double affective = 0.0;         // affective framing (why citation matters to people), [0, 1]
  int outcomes = 0;               // learning outcomes written for the session (0: none)
};

/** Coverage-based: every item, mostly lecture, no stated outcomes. */
SessionPlan coveragePlan(const std::vector<ContentItem>& items, double minutes);
/** Activity-based: an hour of hands-on database searching, not linked to a big idea. */
SessionPlan activityPlan(const std::vector<ContentItem>& items, double minutes);
/**
 * Backward Design with k outcomes: the first k items in priority order (the basic ideas of the
 * first big idea, then that big idea, then the next), time in proportion to what each needs.
 */
SessionPlan backwardDesignPlan(const std::vector<ContentItem>& items, double minutes, int outcomes);
/** Priority order of the items for outcomes: basics then their big idea, big idea by big idea. */
std::vector<int> outcomePriority(const std::vector<ContentItem>& items);

struct SessionParams {
  int learners = 400;
  double aptitudeSd = 0.4;      // sd of e^a's exponent
  double rate = 1.0;            // r
  double capacity = 4.0;        // c: items a session can hold
  double overload = 0.15;       // λ
  double transitionMinutes = 1.0;
  double threshold = 0.6;       // mastery that counts as "met"
};

struct SessionResult {
  std::string plan;
  std::vector<double> mastery;    // mean mastery per item
  int itemsTouched = 0;
  double coverage = 0.0;          // share of items touched
  double basics = 0.0;            // mean mastery of the basic ideas
  double bigIdeas = 0.0;          // mean understanding of the big ideas
  double niceToKnow = 0.0;
  double enduring = 0.0;          // Σ basics + 2 Σ big-idea understanding (the weighted learning)
  double outcomesMet = 0.0;       // share of the plan's outcomes met by the learners (threshold)
  double transfer = 0.0;          // mean transfer of the big ideas to new contexts
  double behaviourChange = 0.0;   // share who use what they learned afterwards
  double selfAssessment = 0.0;    // accuracy of learners' own "did I meet it" judgement
};

SessionResult simulateSession(const std::vector<ContentItem>& items, const SessionPlan& plan, const SessionParams& params,
                              std::uint64_t seed);

/**
 * Outcomes appropriate to the audience: an outcome at Bloom level k has difficulty b₀ + step·k.
 * For an audience of ability θ and the session's scaffold, each level's chance of success alone
 * and with help comes from the ZPD geometry (pure elliptic CGA, social/zpd.hpp). The recommended
 * level is the highest one inside the zone of proximal development.
 */
struct OutcomeLevelFit {
  std::vector<double> alone, assisted, gain;  // per level
  std::vector<int> zone;                      // Zone per level
  int recommended = 0;
};
OutcomeLevelFit fitOutcomeLevel(double audience, double scaffold, double baseDifficulty = -1.5, double step = 0.7);

}  // namespace syn
