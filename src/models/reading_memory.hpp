#pragma once

// What the reader keeps between runs, shared by readingQA, readingStudent, readingNotes and the
// capsule models:
//   - its memory (student_memory.inc): the weights of its reading habits (how to pick an answer
//     to a question), of its note-taking habits (which sentences are worth keeping) and of its
//     constructing habits (how to say an answer from what it knows), learned from golden answers,
//     with a fingerprint of the habits they were learned for;
//   - its knowledge, a capsule (reader_capsule.inc): the passages it read, ingested into the space —
//     each word a multivector whose grades (fact, word point, grammar bivector) an index router
//     routes to packed codes — and a lexicon of the words it knows. Answers are constructed from the
//     capsule by the grammar, never copied from the text.
// Host-side preparation only: parsing, the habits as features of candidates; encoding, recall and
// scoring run in WGSL.
#include <cstdint>
#include <string>
#include <vector>

#include "syn/core/value.hpp"

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
/** The constructing habits: features of an answer the grammar constructs from a recalled fact. */
extern const char* const kConstructHabits[kHabitCount];
/** The priors: the lexical reader (overlap, one sentence), notes of the most informative words,
 *  and saying the whole fact that shares most with the question. */
extern const double kHabitPrior[kHabitCount];
extern const double kNotePrior[kHabitCount];
extern const double kConstructPrior[kHabitCount];

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
  std::string design, source, notesDesign, constructDesign;
  double test = 0, notesTest = 0, constructTest = 0;
  double habits[kHabitCount] = {};
  double notes[kHabitCount] = {};
  double construct[kHabitCount] = {};
};

/** The stored memory, checked against the habits of this build (or `fingerprintOf`, to check a
 *  memory written for other habits). Refused, or with `fresh`, it is the prior. */
Memory recall(bool fresh, const std::string& fingerprintOf = "");

// ---- The grammar and the capsule.

/** A fact the grammar reads in a clause: its subject, relation (the verb and what goes with it)
 *  and object, as words (articles dropped). */
struct Fact {
  std::vector<std::string> subject, relation, object;
};
/** The grammatical roles, in the order a fact is said. */
constexpr int kRoles = 3;
extern const char* const kRoleNames[kRoles];

/** The facts of a passage: its sentences split into clauses, each read as subject, relation, object. */
std::vector<Fact> grammar(const std::string& passage);
/** Where a word lives in the space: three coordinates that `lift` takes to a point of S³ (a hash
 *  of the word, the same for every reader). */
void wordCoordinates(const std::string& word, float u[3]);
/** The grammar as operators, in WGSL: binding a word's point by the rotor of its role and position
 *  (bindWord), and reading the grammar's rotor back from a point and a word's point (grammarOf). */
const char* grammarWgsl();
/** Saying words: joined, the first capitalised, a full stop. */
std::string realize(const std::vector<std::string>& words);

/** The facts recalled from a capsule (capsuleRecall's result), per passage, as words. */
std::vector<std::vector<Fact>> factsOf(const Value& recalled);
/** Answer candidates the grammar constructs from facts (the whole fact, its predicate, its object,
 *  its subject), with the constructing habits as features and their token F1 against the golden
 *  answer. `facts[i]` are the facts item i may answer from. */
std::vector<Candidates> constructCandidates(const std::vector<Item>& items, const std::vector<std::vector<Fact>>& facts);
/** For a question with no capsule: the facts of all capsules that share most with it. */
std::vector<Fact> factsFor(const std::string& question, const std::vector<std::vector<Fact>>& all, std::size_t keep);

/** The capsule packed: the lexicon front-coded, and per passage its vocabulary (level 1's
 *  codebook), its facts as run lengths (level 0) and each slot's code (levels 1 and 2: the word in
 *  the vocabulary, the role and the place), bit-packed. The bytes of each part, and all in base64. */
struct CapsuleBytes {
  std::vector<std::uint8_t> bytes;
  std::size_t lexicon = 0, vocabulary = 0, facts = 0, codes = 0;
  std::string base64;
};
CapsuleBytes packCapsule(const Value& capsule);
/** The capsule from its packed bytes (base64). */
Value unpackCapsule(const std::string& base64);

/** The stored capsule (reader_capsule.inc): the golden passages, ingested; capsuleRecall's default. */
const Value& storedCapsule();

}  // namespace syn::models::memory
