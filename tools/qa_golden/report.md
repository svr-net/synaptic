# QA golden test

200 golden questions (edededdy/fineweb-edu-grounded-qa, FineWeb-Edu passages), 200 passages + 300 knowledge documents read; vocabulary 13184 words, 6894 sentences.

| answer | token F1 | F1 ≥ 0.5 |
|---|---:|---:|
| **reasoned from its capsule alone (capsuleQA: question parts, evidence in the space, answers constructed)** | **0.337** | |
| reader, its own passage (copies a sentence) | 0.343 | 25.5% |
| reader with its remembered habits, its own passage (copies one or two sentences) | 0.374 | 31% |
| constructed from its knowledge capsule alone (never sees the text) | 0.342 | 23% |
| reader, open book (the whole corpus) | 0.34 | 25.5% |
| first sentence of the passage | 0.171 | 4% |
| a random sentence of the passage | 0.154 | 3% |
| oracle: the best sentence of the passage | 0.503 | 46.5% |

Open book, the reader's answer came from the question's own passage for 96% of the questions.

## The knowledge capsule

The grammar read the 600 passages (200 of the test, 400 of the lessons; 327626 words) as 18404 facts — subject, relation, object. Every word of a fact is one multivector of the space whose grades are the levels of the index: grade 0 its fact, grade 1 its word's point, grade 2 the grammar's bivector (role and place). The index router isolates each grade and routes it to its codebook — the passage's vocabulary, the grammar's lattice of 3 roles × 32 places, the runs of facts — and the capsule keeps only the codes, over a lexicon of 29104 words. Recalled from the codes alone — the rotor rebuilt from grade 2, the word bound by it, its role and place read back off the bound point — 100% of the 249064 words came back exactly.

| | bytes |
|---|---:|
| the text | 1985545 |
| **the capsule, packed** | **868089 (43.7% of the text)** |
| codes: a word's place in its vocabulary, role and place (levels 1, 2) | 474030 |
| the lexicon, front-coded | 151904 |
| the vocabularies (level 1's codebooks) | 179478 |
| the facts, as run lengths (level 0) | 19004 |
| the growing index: each new word's cell of S³ | 43656 |
| the segment's header and checksum | 15 |

Constructed answers are said by the grammar from the recalled facts — the whole fact, its predicate, its object, its subject or two facts — and chosen by the constructing habits; the text is never used.

## Reasoning from the capsule

capsuleQA splits a question into its parts (28 of 200 questions have two), finds for each part the evidence — the facts of the capsule that hold the question's informative words, by identity of their points — and weighs the grammar's ways of saying each fact by the constructing habits; each part's answer is one sentence. Memory: partly: constructing habits were learned for other habits; those start from the prior. Three worked answers:

**Q:** What are the three main components used to calculate the ROI for conservation in the described project, and which data sources were used for each?

- part: *What are the three main components used to calculate the ROI for conservation in the described project,*
  - evidence: [project] [described in] [recent article in journal Biological Conservation was first to compare terrestrial freshwater and coastal habitats in], belief 66%
  - reasoning: says the whole fact; weighed by says the whole fact, restating its subject (1); covers the question (0.333); evidence: the fact holds the question's informative words (0.312);
  - answer: Project described in recent article in journal Biological Conservation was first to compare terrestrial freshwater and coastal habitats in.
- part: *which data sources were used for each?*
  - evidence: [It also] [takes] [advantage of new sources of geographic data developed in past decade], belief 81%
  - reasoning: says the whole fact; weighed by says the whole fact, restating its subject (1); evidence: the fact holds the question's informative words (0.787); covers the question (0.667);
  - answer: It also takes advantage of new sources of geographic data developed in past decade.

**Golden:** The three components are benefit, probability of success, and cost. The authors used the Biological Distinctiveness Index (BDI) and habitat condition for benefit; the Ibrahim Index of African Governance (IIAG) for probability of success; and the costs of managing protected areas or displacing agriculture for the cost component. — F1 0.107

**Q:** What happened to the Jewish dealers and their clientele during World War II?

- part: *What happened to the Jewish dealers and their clientele during World War II?*
  - evidence: [substantial number of Jewish art and antiques] [dealers] [operated in pre-World War II Warsaw], belief 85%
  - reasoning: says the fact and the next; weighed by says the whole fact, restating its subject (1); evidence: the fact holds the question's informative words (0.626); covers the question (0.625); says the detail next to it (two facts) (0.5);
  - answer: Substantial number of Jewish art and antiques dealers operated in pre-World War II Warsaw. Particularly respected were salons of brothers Bernard and Abe Gutnajer.

**Golden:** Virtually everyone in the Jewish dealers' milieu perished, with members either perishing in the Warsaw ghetto or in Treblinka. — F1 0.195

**Q:** What is the recommended long-term commitment parents should make regarding reading for their family in 2017?

- part: *What is the recommended long-term commitment parents should make regarding reading for their family in 2017?*
  - evidence: [] [] [However even if you only stick to one resolution this year pledge to make reading year-long commitment for your family], belief 81%
  - reasoning: says the fact and the next; weighed by says the whole fact, restating its subject (1); says the detail next to it (two facts) (0.5); covers the question (0.5); evidence: the fact holds the question's informative words (0.463);
  - answer: However even if you only stick to one resolution this year pledge to make reading year-long commitment for your family. Ability to read is precious commodity.

**Golden:** Parents should pledge to make reading a year-long commitment for their family. Even if they can only stick to one resolution, prioritizing reading ensures children benefit from consistent encouragement and exposure to literacy. — F1 0.552

## The reader as a student: picking an answer

Candidates: one sentence of the passage or two adjacent ones. 400 lessons (golden questions about other passages), three passes, 200 test questions, 64 students per design; the best candidate scores 0.543. Its memory: generation 4, 4800 lessons studied before.

| study design | test F1 | lessons F1 |
|---|---:|---:|
| no feedback (the lexical reader) | 0.334 | 0.354 |
| its memory alone, no feedback | 0.374 | 0.391 |
| reinforcement | 0.362 | 0.388 |
| reinforcement + schema: starts from its memory | 0.365 | 0.405 |
| reinforcement + interleaving question kinds | 0.363 | 0.389 |
| reinforcement + worked examples | 0.372 | 0.395 |
| reinforcement + spaced retrieval of mistakes | 0.36 | 0.389 |
| reinforcement + equilibration: assimilate or accommodate | 0.363 | 0.386 |
| reinforcement + scaffolding: easy before hard | 0.362 | 0.389 |
| reinforcement + peer discussion | 0.362 | 0.381 |
| reinforcement + self-direction | 0.362 | 0.388 |
| reinforcement + growth mindset | 0.362 | 0.388 |
| reinforcement + reflection: one or two changes at a time | 0.366 | 0.385 |
| all methods | 0.356 | 0.376 |
| all methods but schema: starts from its memory | 0.352 | 0.369 |
| all methods but interleaving question kinds | 0.363 | 0.383 |
| all methods but worked examples | 0.369 | 0.391 |
| all methods but spaced retrieval of mistakes | 0.352 | 0.37 |
| all methods but equilibration: assimilate or accommodate | 0.358 | 0.375 |
| all methods but scaffolding: easy before hard | 0.36 | 0.382 |
| all methods but peer discussion | 0.362 | 0.379 |
| all methods but self-direction | 0.356 | 0.375 |
| all methods but growth mindset | 0.356 | 0.377 |
| all methods but reflection: one or two changes at a time | 0.369 | 0.39 |

| method | library models | added to reinforcement | left out of all |
|---|---|---:|---:|
| schema: starts from its memory | schema | +0.003 | -0.004 |
| interleaving question kinds | interleaving | +0.001 | +0.007 |
| worked examples | gagne, lecture | +0.011 | +0.014 |
| spaced retrieval of mistakes | memory | -0.002 | -0.004 |
| equilibration: assimilate or accommodate | equilibration | +0.001 | +0.002 |
| scaffolding: easy before hard | zpd, scaffolding | +0.001 | +0.004 |
| peer discussion | dialogue, discussion | -0 | +0.006 |
| self-direction | andragogy | +0 | +0 |
| growth mindset | motivation | +0.001 | -0 |
| reflection: one or two changes at a time | reflection | +0.004 | +0.013 |

## The reader as a student: constructing an answer from its capsule

Candidates: what the grammar can say from each fact recalled from the capsule — the whole fact, its predicate, its object, its subject. 400 lessons (golden questions about other passages), three passes, 200 test questions, 64 students per design; the best candidate scores 0.505. Its memory: generation 4, 4800 lessons studied before.

| study design | test F1 | lessons F1 |
|---|---:|---:|
| no feedback (the lexical reader) | 0.342 | 0.357 |
| its memory alone, no feedback | 0.342 | 0.357 |
| reinforcement | 0.342 | 0.358 |
| reinforcement + schema: starts from its memory | 0.342 | 0.358 |
| reinforcement + interleaving question kinds | 0.342 | 0.358 |
| reinforcement + worked examples | 0.34 | 0.361 |
| reinforcement + spaced retrieval of mistakes | 0.343 | 0.357 |
| reinforcement + equilibration: assimilate or accommodate | 0.343 | 0.358 |
| reinforcement + scaffolding: easy before hard | 0.342 | 0.358 |
| reinforcement + peer discussion | 0.342 | 0.358 |
| reinforcement + self-direction | 0.342 | 0.358 |
| reinforcement + growth mindset | 0.342 | 0.358 |
| reinforcement + reflection: one or two changes at a time | 0.342 | 0.358 |
| all methods | 0.339 | 0.358 |
| all methods but schema: starts from its memory | 0.339 | 0.358 |
| all methods but interleaving question kinds | 0.341 | 0.361 |
| all methods but worked examples | 0.342 | 0.358 |
| all methods but spaced retrieval of mistakes | 0.337 | 0.354 |
| all methods but equilibration: assimilate or accommodate | 0.34 | 0.358 |
| all methods but scaffolding: easy before hard | 0.339 | 0.358 |
| all methods but peer discussion | 0.336 | 0.356 |
| all methods but self-direction | 0.34 | 0.359 |
| all methods but growth mindset | 0.34 | 0.358 |
| all methods but reflection: one or two changes at a time | 0.341 | 0.36 |

| method | library models | added to reinforcement | left out of all |
|---|---|---:|---:|
| schema: starts from its memory | schema | +0 | +-0 |
| interleaving question kinds | interleaving | +0 | +0.002 |
| worked examples | gagne, lecture | -0.002 | +0.003 |
| spaced retrieval of mistakes | memory | +0 | -0.002 |
| equilibration: assimilate or accommodate | equilibration | +0 | +0.001 |
| scaffolding: easy before hard | zpd, scaffolding | +0 | -0 |
| peer discussion | dialogue, discussion | +0 | -0.003 |
| self-direction | andragogy | +0 | +0.001 |
| growth mindset | motivation | +0 | +0.001 |
| reflection: one or two changes at a time | reflection | +0 | +0.002 |

## The reader as a student: taking notes

Candidates: the sentences of a passage, before any question. 400 lessons (golden questions about other passages), three passes, 200 test questions, 64 students per design; the best candidate scores 0.503. Its memory: generation 4, 4800 lessons studied before.

| study design | test F1 | lessons F1 |
|---|---:|---:|
| no feedback (the lexical reader) | 0.198 | 0.2 |
| its memory alone, no feedback | 0.196 | 0.201 |
| reinforcement | 0.201 | 0.208 |
| reinforcement + schema: starts from its memory | 0.199 | 0.205 |
| reinforcement + interleaving question kinds | 0.197 | 0.205 |
| reinforcement + worked examples | 0.192 | 0.206 |
| reinforcement + spaced retrieval of mistakes | 0.202 | 0.217 |
| reinforcement + equilibration: assimilate or accommodate | 0.2 | 0.214 |
| reinforcement + scaffolding: easy before hard | 0.197 | 0.203 |
| reinforcement + peer discussion | 0.197 | 0.203 |
| reinforcement + self-direction | 0.197 | 0.205 |
| reinforcement + growth mindset | 0.206 | 0.208 |
| reinforcement + reflection: one or two changes at a time | 0.196 | 0.203 |
| all methods | 0.188 | 0.189 |
| all methods but schema: starts from its memory | 0.188 | 0.189 |
| all methods but interleaving question kinds | 0.184 | 0.184 |
| all methods but worked examples | 0.198 | 0.206 |
| all methods but spaced retrieval of mistakes | 0.193 | 0.196 |
| all methods but equilibration: assimilate or accommodate | 0.188 | 0.187 |
| all methods but scaffolding: easy before hard | 0.182 | 0.182 |
| all methods but peer discussion | 0.162 | 0.155 |
| all methods but self-direction | 0.182 | 0.182 |
| all methods but growth mindset | 0.187 | 0.187 |
| all methods but reflection: one or two changes at a time | 0.196 | 0.199 |

| method | library models | added to reinforcement | left out of all |
|---|---|---:|---:|
| schema: starts from its memory | schema | -0.002 | +0 |
| interleaving question kinds | interleaving | -0.004 | -0.004 |
| worked examples | gagne, lecture | -0.009 | +0.01 |
| spaced retrieval of mistakes | memory | +0.001 | +0.005 |
| equilibration: assimilate or accommodate | equilibration | -0.001 | +0 |
| scaffolding: easy before hard | zpd, scaffolding | -0.004 | -0.006 |
| peer discussion | dialogue, discussion | -0.003 | -0.026 |
| self-direction | andragogy | -0.003 | -0.006 |
| growth mindset | motivation | +0.005 | -0.001 |
| reflection: one or two changes at a time | reflection | -0.005 | +0.008 |

Its next memory (generation 5): reading, note-taking and constructing habits learned with all methods.

## What it keeps, to grow

| segment | passages | words new to it | words it knew | bytes | text bytes | routed |
|---|---:|---:|---:|---:|---:|---:|
| 000000 (the stored capsule) | 60 | 6987 | 0 | 115560 | 211733 | 1 |
| 000001 (the knowledge corpus, on top) | 300 | 5238 | 6987 | 137912 | 266661 | 1 |

Recalled from both segments: 360 passages, 100% of the words exactly.

The records of this run's studies: what each would make of the memory — its weights folded in by their evidence — answering the test (the reading habits: remembering; the constructing habits: constructing). A record is kept only when the memory with it answers at least as well as without it.

| record | generation | evidence (lessons) | its study alone | memory | memory with it | kept |
|---|---:|---:|---:|---:|---:|---|
| answer.rec | 5 | 6000 | 0.356 | 0.374 | 0.373 | not kept: the memory answers worse with it |
| notes.rec | 5 | 6000 | 0.188 | – | – | it answers worse than its memory alone: not kept |
| construct.rec | 5 | 1200 | 0.339 | 0.342 | 0.341 | not kept: the memory answers worse with it |

Written to /tmp/claude-0/reader.
