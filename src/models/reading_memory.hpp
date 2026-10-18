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
  double evidence[3] = {};   // lessons behind each section's weights (reading, notes, constructing)
  Value records = Value::array();  // the records read on top of the stored memory: what each added, or why not
};

/** The stored memory, checked against the habits of this build (or `fingerprintOf`, to check a
 *  memory written for other habits). Refused, or with `fresh`, it is the prior. */
Memory recall(bool fresh, const std::string& fingerprintOf = "", const Value& records = Value());
/** The memory a model's spec asks for: `memory` (false: fresh), `memoryFingerprint`, and
 *  `memoryRecords` — records read on top of the stored memory, in order. */
Memory recallFor(const Value& spec);

/** The sections of the memory, in the order of Memory::evidence. */
constexpr int kSections = 3;
extern const char* const kSectionNames[kSections];  // "answer", "notes", "construct"
/** A memory record: what one study added to one section — the weights it learned from `lessons`
 *  lessons, and its test score — for the generation it makes. Memory grows by appending records: each
 *  is folded into the section by its evidence, (N·w + n·w_b) / (N + n), so a small study moves a well
 *  learned habit a little and a fresh one all the way. Packed (magic, version, the section's
 *  fingerprint, generation, lessons, test, weights, design, source, checksum) in base64. */
struct Record {
  int section = 0, generation = 0;
  double lessons = 0, test = 0;
  double weights[kHabitCount] = {};
  std::string design, source;
};
std::string packRecord(const Record& r);
/** A record read back; empty `error` when it can be used with these habits. */
Record unpackRecord(const std::string& base64, std::string* error);
/** The weights a section would have with a record folded in. */
void fold(const double* weights, double evidence, const Record& r, double* out);

// ---- The grammar and the capsule.

/** A fact the grammar reads in a clause: its subject, relation (the verb and what goes with it)
 *  and object, as words (articles dropped). */
struct Fact {
  std::vector<std::string> subject, relation, object;
};
/** The grammatical roles, in the order a fact is said. */
constexpr int kRoles = 3;
extern const char* const kRoleNames[kRoles];

/** The cues a reader knows of a word on its own (bits): its word class's closed lists, its ending,
 *  its capital, its neighbour. Which word of a clause is the verb, and so every word's role, is
 *  decided in the space (capsuleIngest): the cues dilate a point, the verb class is a cap. */
constexpr std::uint32_t kCueAuxiliary = 1, kCueVerbEnding = 2, kCueFunctionWord = 4, kCueCapital = 8, kCueAfterPreposition = 16,
                        kCueFirst = 32, kCueGoesWithVerb = 64, kCueAfterPlural = 128;
/** A clause of a passage (at least three words, articles dropped) and the cues of its words. */
struct Clause {
  std::vector<std::string> words;
  std::vector<std::uint32_t> cues;
};
/** The clauses of a passage: its sentences split at ; : and at a comma before a conjunction. */
std::vector<Clause> clauses(const std::string& passage);
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
/** A part of a question (one sentence of the answer each): its words, and the words it refers back
 *  to ("their", "they", "it" → the first part's subject). */
struct QuestionPart {
  std::string text;
  std::vector<std::string> stems;     // its informative words (stemmed)
  std::vector<std::string> referred;  // what its pronouns refer to, from the part before
  bool asksNumber = false, asksName = false;
};
/** The parts of a question: split before a second question word ("What are X and what are their Y?"). */
std::vector<QuestionPart> questionParts(const std::string& question);
/** The informative words (stems) a word of the lexicon carries. */
std::vector<std::string> stemsOf(const std::string& word);

/** For a question with no capsule: the facts of all capsules that share most with it. */
std::vector<Fact> factsFor(const std::string& question, const std::vector<std::vector<Fact>>& all, std::size_t keep);

/** The capsule packed, segment by segment. A capsule grows by appending segments: each adds words to
 *  the lexicon (numbered after the words before it, so no index ever changes) and passages, and is
 *  packed on its own — a header (magic, version, the capsule's fingerprint, the lexicon and passages
 *  it extends), its new words front-coded with the cell of the index each lies in, per passage its
 *  vocabulary (level 1's codebook), its facts as run lengths (level 0) and each slot's code (levels
 *  1 and 2: the word in the vocabulary, the role and the place), bit-packed, and a checksum. The
 *  bytes of each part, and all in base64. */
struct CapsuleBytes {
  std::vector<std::uint8_t> bytes;
  std::size_t header = 0, lexicon = 0, cells = 0, vocabulary = 0, facts = 0, codes = 0;
  std::string base64;
};
/** The fingerprint of the capsule's format: the word points, the grammar, the codes and the index. A
 *  segment written for another is refused, never misread. */
std::uint32_t capsuleFingerprint();
/** The segment of a capsule that extends its first `baseWords` words and `basePassages` passages. */
CapsuleBytes packSegment(const Value& capsule, std::size_t baseWords, std::size_t basePassages);
/** The whole capsule as one segment. */
CapsuleBytes packCapsule(const Value& capsule);
/** Segments read in order into one capsule ({dictionary, cells, passages}), with a report per segment
 *  (bytes, words, passages, used, and why not). A segment that is damaged, written for another
 *  format, or does not extend the capsule read so far is refused, and so are those after it. */
Value joinSegments(const std::vector<std::string>& segments);
/** One segment, read as a capsule. */
Value unpackCapsule(const std::string& base64);

/** The stored capsule (reader_capsule.inc): the golden passages, ingested; capsuleRecall's default. */
const std::vector<std::string>& storedSegments();
const Value& storedCapsule();
/** The capsule a model's spec names: `capsule` as is; `segments` (or one, `packed`) read in order —
 *  on top of the stored capsule when they extend it (or with `stored`); none: the stored capsule.
 *  {capsule, segments (the report), stored}. */
Value capsuleOf(const Value& spec);

}  // namespace syn::models::memory
