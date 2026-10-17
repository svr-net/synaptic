#pragma once

// What the reader keeps between runs, shared by readingQA, readingStudent and readingNotes:
//   - its memory (student_memory.inc): the weights of its reading habits (how to pick an answer
//     to a question) and of its note-taking habits (which sentences are worth keeping), learned
//     from golden answers, with a fingerprint of the habits they were learned for;
//   - its knowledge (reader_knowledge.inc): the notes it took on the passages it read, so that it
//     can answer when the passages are gone.
// Host-side preparation only: the habits are features of candidates; scoring them runs in WGSL.
#include <string>
#include <vector>

namespace syn::models::memory {

constexpr int kHabitCount = 8;

/** A question about a passage, with its golden answer when known. */
struct Item {
  std::string question, golden, passage;
};

/** The candidates of one item, their habit features, their token F1 against the golden answer,
 *  and the best of them. */
struct Candidates {
  std::vector<std::string> texts;
  std::vector<std::vector<double>> habits;
  std::vector<double> reward;
  std::vector<bool> pair;
  std::size_t best = 0;
};

/** The reading habits: features of an answer (one sentence of the passage or two adjacent ones)
 *  to a question. */
extern const char* const kHabits[kHabitCount];
/** The note-taking habits: features of a sentence of a passage, without a question. */
extern const char* const kNoteHabits[kHabitCount];
/** The priors: the lexical reader (overlap, one sentence), and notes of the most informative words. */
extern const double kHabitPrior[kHabitCount];
extern const double kNotePrior[kHabitCount];

/** The fingerprint of both habit sets: a memory is used only for the habits it was learned for. */
std::string fingerprint();

/** Answer candidates of each item: its passage's sentences, then adjacent pairs. */
std::vector<Candidates> answerCandidates(const std::vector<Item>& items);
/** Note candidates of each item: its passage's sentences (no pairs), with question-free features. */
std::vector<Candidates> noteCandidates(const std::vector<Item>& items);

/** The memory, as the reader may use it. */
struct Memory {
  bool used = false;        // false: fresh, or the stored memory was refused
  std::string reason;       // why it was not used
  int generation = 0;
  double lessons = 0;
  std::string design, source, notesDesign;
  double test = 0, notesTest = 0;
  double habits[kHabitCount] = {};
  double notes[kHabitCount] = {};
};

/** The stored memory, checked against the habits of this build (or `fingerprintOf`, to check a
 *  memory written for other habits). Refused, or with `fresh`, it is the prior. */
Memory recall(bool fresh, const std::string& fingerprintOf = "");

/** The knowledge: the notes on each passage the reader read (in the golden set's order). */
struct Knowledge {
  std::vector<std::string> sources;  // the passage each note came from
  std::vector<std::string> notes;
  double kept = 0;                   // the share of the passages' text kept
  int perPassage = 0;
  int generation = 0;                // of the memory that took the notes
};
const Knowledge& knowledge();

}  // namespace syn::models::memory
