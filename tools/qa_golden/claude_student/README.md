# Claude as the student

Claude studied the QA golden lessons the way `readingStudent` does, and its habits became the
model's constructing prior (`kConstructPrior` in `src/models/reading_memory.cpp`), used by
`capsuleQA` and by the student's construct mode.

- **Lessons**: 15 golden questions about other passages (the first of `train.tsv`), three blocks of
  five. Each answer was worked out (evidence → reasoning → answer) before its golden answer was
  seen, scored with SQuAD token F1 (as `qa_golden`), and studied: reinforcement, worked examples,
  spaced retrieval of the worst lesson, reflection with one or two changes per block
  (`journal.md`, `lessons_block{1,2,3}.json`).
- **Habits** (`habits.md`): answer in full sentences restating the question's subject; keep the
  detail right next to the fact; name two or three examples and who the passage credits; one
  sentence per part of the question, two or three in all.
- **Tests**: the first 30 questions of the golden test, answered by a fresh Claude cold
  (`pretest_answers.json`) and by another given only the habits (`posttest_answers.json`); neither
  saw the golden answers, the lessons or the journal.

| | token F1, 30 test questions |
|---|---:|
| the library's reader (copies a sentence) | 0.315 |
| the library's remembered habits | 0.342 |
| constructed from the capsule | 0.309 |
| best single sentence (oracle) | 0.503 |
| Claude, before the lessons | 0.489 |
| **Claude, after the lessons (habits only)** | **0.618** |

26 of 30 answers improved. The gain is mostly answer form — length and wording matching how the
golden answers are written — on a small sample (15 lessons, 30 questions, one run each).

In the model, the habits map onto the constructing features: evidence (the fact holds the
question's informative words), saying the whole fact (restating its subject), the detail next to
it (two facts), holding what is asked, covering the question; and "one sentence per part" is
capsuleQA's question decomposition.
